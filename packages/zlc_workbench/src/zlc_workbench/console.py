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
from operator import attrgetter
from pathlib import Path
from queue import Empty, SimpleQueue
import time
from typing import Any

from zlc_plot import (
    accepts_classifier_thresholds,
    DEFAULTS,
    describe_semantics,
    IMAGE_POINT_OVERLAY_CONTRACT,
    IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
    history_window_requirement,
    PlotKind,
    SelectorKind,
    paints_image_surface,
    image_point_overlay_from_signal,
)
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointStatus
from zlc_plot.semantics import SemanticVacancy
from zlc_plot.specs import semantic_spec, validate_authored_display
from zlc_plot.ui import parameter_controls_for_kind
from zlc_runtime import (
    DatasetCoverage,
    IndexedHistoryLease,
    OperatorInputRequest,
    SelectionChange,
)
from zlc_ui import FormFieldProps, FormSpec

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
    LogicDraftFinalization,
    artifact_input_specs,
    build_arguments,
    dataset_inputs,
    device_key_options,
    finalize_logic_draft,
    make_host,
    stable_signal_key,
    task_input_summary,
)
from .panel_save import (
    capture_run_chain,
    save_panel_figure as _save_panel_figure,
)
from .panel_catalog import (
    GRID_CELL_KINDS,
    TASK_CONSOLE_PANEL_CATALOG,
    panel_kind_choices,
    task_console_fitting_spec,
    task_console_panel_identity,
    task_console_panel_kind,
)
from .panel_state import (
    PanelFrozenData,
    PanelState,
    control_document,
    fit_output_fields,
    fit_edit_targets,
    panel_state_from_description,
    panel_data_shape,
    panel_surface_from_description,
    project_panel_state,
    semantic_entries,
)
from .presentation import PlotPanelPort
from .selection import (
    PlotSelectionSource,
    panel_selection_document,
    panel_selection_derives_signal,
    panel_selection_from_document,
    panel_selection_matches_subject,
    panel_selection_output_catalog,
    panel_plot_selectors,
    observation_matches_plot_input,
    plot_identity_matches_plot_input,
    _apply_panel_selection,
    _remove_panel_selection,
    attach_selection_bridge,
)
from .topology import format_signal_shape, project_signals

__all__ = ["ConsolePresenter", "PanelBinding", "PanelState"]


_UNCHANGED = object()


def _run_inline(work, deliver, failed) -> None:
    try:
        deliver(work())
    except BaseException as error:
        failed(error)


def _error_text(error: BaseException) -> str:
    """The message an exception carries, or its class name when it has none.

    ``CancelledError``, a bare ``assert`` and ``TimeoutError`` all stringify
    to nothing, and ``f"{title}: {error}"`` then put a red line on the status
    strip with nothing after the colon.
    """

    return str(error) or type(error).__name__


def _same_panel_selection(left: object, right: object) -> bool:
    def signature(selection: object) -> tuple[object, ...]:
        return (
            selection.plot_kind,
            selection.selector_kind,
            selection.ranges,
            selection.facets,
            selection.repeat_index,
        )

    return signature(left) == signature(right)


_PLOT_TARGET = attrgetter(
    "signal", "kind", "cell_kind", "size", "semantic", "display", "fit",
    "overlay_signal", "selector", "classifier_thresholds", "focused_cell",
)


def _same_panel_plot_target(left: PanelState, right: PanelState) -> bool:
    return _PLOT_TARGET(left) == _PLOT_TARGET(right)


def _run_of(publication: object) -> object | None:
    """Which RUN a publication belongs to.

    The generation IS the run: everything published under it came out of one
    execution of the node.  A publication carrying none answers for itself,
    so two objects are two moments either way.
    """

    generation = getattr(getattr(publication, "event_ref", None), "generation", None)
    return publication if generation is None else generation


@dataclass
class PanelBinding:
    """One runtime binding around the panel's single authored state."""

    panel_id: str
    state: PanelState
    port: PlotPanelPort | None = None
    #: Runtime-owned source-index retention exists only while this panel asks
    #: for a history window.  The lease, not the signal declaration, is the
    #: resource owner.
    history_lease: IndexedHistoryLease | None = None
    #: Edit deliberately keeps one frozen data revision until Refresh.  It is
    #: not panel configuration and therefore does not live in ``PanelState``.
    frozen_data: PanelFrozenData | None = None
    #: Refresh waits for the next surface the live port actually accepts.
    #: Until then Edit/Save keep the previous accepted snapshot.
    refresh_requested: bool = False
    #: Panel Edit owns a second, frozen plotting surface.  It is deliberately
    #: not the live monitor host and therefore has no PlotPanelPort.
    editor_host: Any = None
    editor_selections: Any = None
    editor_open: bool = False
    #: Live derivation from selections drawn on this panel, if it has one.
    bridge: Any = None
    selections: Any = None
    #: The panel's own record holds the selector, the classifier level and the
    #: opened cell -- they are saved with the board.  Only the VIEWPORT stays
    #: here: a zoom is measured in display coordinates of one exact picture,
    #: and pasting yesterday's numbers onto today's data is how a panel opens
    #: showing a range nobody chose.
    #: The display viewport shared by both views, and the data it was
    #: measured on.  A range means nothing once the axes underneath it change,
    #: so it is kept beside the publication it was taken from and dropped when
    #: that no longer describes what is on screen.
    interaction_viewport: Any = None
    #: The last failure already shown, so one refusal is reported once.
    reported_error: Any = None
    #: Image annotation has its own presentation revision while its data remains
    #: an explicit sibling in the selected publication.
    overlay_revision: int = -1
    #: UI-neutral parameter descriptions projected from this host's public
    #: zlc_plot control plane.  This is editor metadata, not a second authored
    #: state; accepted values still live only in ``state``.
    parameter_surface: Mapping[str, object] = field(default_factory=dict)
    #: Monotonic across both Live and Edit hosts.  Plot selector revisions are
    #: host-local, so two surfaces can both emit revision 1; Runtime bridge
    #: triggers need the panel's single canonical sequence instead.
    selection_revision: int = 0
    configuration: Any = None
    editor_configuration: Any = None

    @property
    def accepted_surface(self) -> object | None:
        return None if self.port is None else self.port.accepted_surface()

    @property
    def host(self) -> Any:
        surface = self.accepted_surface
        return None if surface is None else surface.host

    @property
    def display_publication(self) -> Any:
        surface = self.accepted_surface
        return None if surface is None else surface.publication

    @property
    def accepted_display(self) -> object | None:
        surface = self.accepted_surface
        return None if surface is None else surface.description

    @property
    def frozen_stale(self) -> bool:
        """Whether Edit's frozen picture still describes what the bench holds.

        A comparison between two moments this binding already holds: what
        Edit froze, and what the card is showing.  Another signal, or the
        same signal from a later RUN, means the frozen fit was solved against
        data the bench no longer has.

        As a stored boolean it needed a writer everywhere either moment could
        change and a rollback in the one place that could fail -- and it was
        still missed at the one that mattered, which is how the Edit tab sat
        on the previous run's picture with nothing saying so.  Derived, it
        cannot be forgotten.
        """

        frozen = self.frozen_data
        if frozen is None:
            return False
        fields = (
            "signal",
            "kind",
            "cell_kind",
            "size",
            "semantic",
            "display",
            "fit",
            "overlay_signal",
        )
        if any(
            getattr(frozen.target, name) != getattr(self.state, name)
            for name in fields
        ):
            return True
        # ``display_publication`` is what this card is showing: the publication
        # its host was built from, and thereafter every one presented on it.
        # Asking the port instead reads one beat behind -- at the moment a new
        # run replaces the host, the port still answers with the old run and
        # the stale mark would arrive a beat late, which is the whole defect.
        shown = self.display_publication
        if shown is None:
            return False
        if _run_of(frozen.publication) != _run_of(shown):
            return True
        # Same run is not the whole story: an EXACT dataset GROWS point
        # by point (a seamless scan commits one readout at a time), and a
        # picture frozen at partial coverage is one column pretending to
        # be the scan.  Growth is staleness too -- comparing run identity
        # alone kept the badge dark while the live card filled in.
        frozen_coverage = getattr(frozen.publication, "coverage", None)
        shown_coverage = getattr(shown, "coverage", None)
        return (
            isinstance(frozen_coverage, DatasetCoverage)
            and isinstance(shown_coverage, DatasetCoverage)
            and shown_coverage.written_cells > frozen_coverage.written_cells
        )

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

    CLOSE_REPORT_SECONDS = 10.0

    def __init__(
        self,
        session: object,
        view: object,
        *,
        make_host: Callable[[object, PanelState], Any],
        spec_for: Callable[[object, str, str], Any] | None = None,
        open_saved: Callable[[str], object] | None = None,
        request_close: Callable[[], None] | None = None,
        run_off_thread: Callable[..., None] | None = None,
        close_worker: Callable[[], bool] | None = None,
        review_points: Callable[[Any, ImagePointOverlay, OperatorInputRequest], object]
        | None = None,
    ) -> None:
        if request_close is not None and not callable(request_close):
            raise TypeError("request_close must be callable or None")
        if run_off_thread is not None and not callable(run_off_thread):
            raise TypeError("run_off_thread must be callable or None")
        if close_worker is not None and not callable(close_worker):
            raise TypeError("close_worker must be callable or None")
        if review_points is not None and not callable(review_points):
            raise TypeError("review_points must be callable or None")
        self.session = session
        self.view = view
        self._make_host = make_host
        # What kinds of panel exist, and whether one dataset admits one.  Both
        # belong to the plotting package; this only asks.
        self._spec_probe = spec_for
        # Reading a saved run is a different window over a different subject,
        # so the console asks for it rather than growing one.
        self._open_saved = open_saved
        self._request_close = request_close
        self._run_off_thread = _run_inline if run_off_thread is None else run_off_thread
        self._close_worker = (lambda: True) if close_worker is None else close_worker
        self._review_points = review_points
        self.logic: dict[str, LogicBinding] = {}
        self.catalog = LogicCatalog()
        # Task identity is a command-admission projection only.  Its lifecycle,
        # phase and progress continue to come exclusively from the row's host.
        self._active_task_id: str | None = None
        self._shown_task_takeover: bool | None = None
        # Auto-created Task previews are reconciled against Runtime terminal
        # truth: an absent signal retires, a sealed signal remains ordinary.
        self._auto_task_previews: dict[str, dict[str, str]] = {}
        self._preview_errors: dict[str, set[str]] = {}
        self._artifact_completion_order = 0
        self.panels: dict[str, PanelBinding] = {}
        # Raster callbacks run on the plot worker.  Their translated, immutable
        # values cross here and are applied only by the existing GUI beat.
        self._panel_interactions: SimpleQueue[Callable[[], None]] = SimpleQueue()
        # Superseded raster hosts finish their current worker task without
        # making the Qt callback wait.  The presenter remains their owner and
        # joins any that are still retiring when the console closes.
        self._retired_plot_hosts: list[Any] = []
        #: Monotonic, so a panel id is never handed out twice in one session.
        self._panel_serial = 0
        # What every card's picker was last told, so it is only rebuilt when
        # the offer really changed.
        self._offered_groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
        self._shown_panel_publishers: tuple[
            tuple[str, tuple[tuple[str, str, str], ...]], ...
        ] = ()
        self._paused = False
        self._deriving = False
        self._shown_console_summary: str | None = None
        self._closing = False
        self._closed = False
        self._close_started_at: float | None = None
        self._close_wait_reported = False
        self._close_retry_sent = False
        self._saving_panels: set[str] = set()
        #: How often a new panel redraws.  The board's default, kept so a panel
        #: and the card that reports it cannot state different numbers.
        live_policy = DEFAULTS.live
        layout_policy = DEFAULTS.layout
        self._default_interval_ms = live_policy.default_refresh_interval_ms

        #: The nine panel presets zlc_plot declares.  A panel records only a
        #: size the plot layer can mount: a state that holds any other string
        #: is a panel whose NEXT mount raises "unknown panel preset", and it
        #: can then never be retargeted at another signal.
        self._sizes = tuple(layout_policy.size_names)
        size_setter = getattr(self.view, "set_panel_sizes", None)
        if callable(size_setter):
            size_setter(layout_policy.size_names, layout_policy.default_preset)

        kinds = panel_kind_choices()
        self._panel_kind_definitions = {
            entry.key: entry for entry in TASK_CONSOLE_PANEL_CATALOG
        }
        self._panel_kind_labels = {
            str(key): str(label or key) for key, label in kinds
        }
        self._default_panel_kind = kinds[0][0] if kinds else ""
        setter = getattr(self.view, "set_panel_kinds", None)
        if setter is not None:
            setter(kinds, self._default_panel_kind)
        cell_setter = getattr(self.view, "set_grid_cell_kinds", None)
        if callable(cell_setter):
            # A FacetGrid's cell kind is a panel PARAMETER: the settings
            # control offers this vocabulary, with empty = the data decides.
            cell_setter(tuple(kind.value for kind in GRID_CELL_KINDS))
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
            intervals=live_policy.refresh_intervals_ms,
        )
        self._intervals = self.board.intervals
        interval_setter = getattr(self.view, "set_panel_intervals", None)
        if callable(interval_setter):
            interval_setter(self._intervals, self._default_interval_ms)
        self._connect()
        self._project_task_takeover()

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
        self.view.panel_edit_requested.connect(self.edit_panel)
        self.view.logic_start_requested.connect(self.start_logic)
        self.view.logic_auto_preview_changed.connect(self.set_logic_auto_preview)
        self.view.logic_stop_requested.connect(self.stop_logic)
        self.view.logic_edit_requested.connect(self.edit_logic)
        self.view.logic_remove_requested.connect(self.remove_logic)
        self.view.stop_task_requested.connect(self.stop_active_task)
        plot_error = getattr(self.view, "panel_plot_error", None)
        if plot_error is not None:
            # The plot widget's own refusal channel (errorOccurred, relayed by
            # the card).  Unconnected, a pointer currency-guard refusal --
            # "the painted pointer front is no longer layout-compatible" --
            # was fully silent and the panel just stopped answering gestures.
            plot_error.connect(self._panel_plot_error)
        draft_changed = getattr(self.view, "logic_draft_changed", None)
        if draft_changed is not None:
            draft_changed.connect(self._logic_draft_changed)
        publisher_edit = getattr(self.view, "panel_publisher_edit_requested", None)
        if publisher_edit is not None:
            publisher_edit.connect(self.edit_panel_publisher)
        publisher_changed = getattr(self.view, "panel_publisher_draft_changed", None)
        if publisher_changed is not None:
            publisher_changed.connect(self._panel_publisher_draft_changed)
        panel_changed = getattr(self.view, "panel_state_changed", None)
        if panel_changed is not None:
            panel_changed.connect(self.update_panel_state)
        refresh_requested = getattr(
            self.view, "panel_snapshot_refresh_requested", None
        )
        if refresh_requested is not None:
            refresh_requested.connect(self.refresh_panel_snapshot)
        producer_restart = getattr(self.view, "panel_producer_restart_requested", None)
        if producer_restart is not None:
            producer_restart.connect(self.restart_panel_producer)
        save_figure = getattr(self.view, "panel_save_figure_requested", None)
        if save_figure is not None:
            save_figure.connect(self.save_panel_figure)
        editor_closed = getattr(self.view, "panel_editor_closed", None)
        if editor_closed is not None:
            editor_closed.connect(self._panel_editor_closed)
        self.set_paused(False)
        self.set_deriving(False)

    # ------------------------------------------------------------------ panels

    def _panel_interval(self, value: object) -> int:
        """Validate against the scheduler policy before state can hold it."""

        normalized = int(value)
        if normalized not in self._intervals:
            raise ValueError(
                f"display interval {normalized} is not in {self._intervals}"
            )
        return normalized

    def _panel_size(self, value: object) -> str:
        """Validate against the plot layer's presets before state can hold it."""

        normalized = str(value)
        if normalized not in self._sizes:
            raise ValueError(f"panel size {normalized!r} is not in {self._sizes}")
        return normalized

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
        overlay_signal: str = "",
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
        definition = self._panel_kind_definitions[wanted]

        try:
            selected_interval = self._panel_interval(
                self._default_interval_ms if interval_ms is None else interval_ms
            )
        except (TypeError, ValueError) as error:
            self._report(_error_text(error), severity="warning")
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
            kind=definition.kind.value,
            # Empty cell kind: the DATA decides.  The panel's settings offer
            # the explicit choice; the Add menu never composes one.
            cell_kind="",
            size=str(size or "2x2"),
            interval_ms=selected_interval,
            title=str(title).strip() or generated_title,
            semantic=dict(semantic or {}),
            display=dict(display or {}),
            fit=dict(fit or {}),
            overlay_signal=str(overlay_signal),
        )
        binding = PanelBinding(
            panel_id,
            state,
            parameter_surface=self._unbound_panel_parameters(state),
        )
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
        overlay_signal: str = "",
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
        signal_name = str(signal)
        front = self.session.signal_plane.freeze()
        current = front.value(signal_name)
        publication = initial_publication
        initial_ref = getattr(initial, "ref", None)
        current_ref = getattr(getattr(current, "snapshot", None), "ref", None)
        if publication is None and current is not None and (
            current_ref == initial_ref
            or (
                getattr(current_ref, "revision", None)
                == getattr(initial_ref, "revision", None)
                and getattr(current_ref, "stream_generation", None)
                == getattr(initial_ref, "stream_generation", None)
            )
        ):
            publication = front.publication(signal_name)
        exact_value = self._publication_value(publication, signal_name)

        wanted = str(kind)
        if not wanted:
            inferred = (
                self._spec_for(initial, "")
                if exact_value is None
                else self._spec_for_value(exact_value, "")
            )
            inferred_kind = getattr(getattr(inferred, "kind", None), "value", None)
            if not inferred_kind:
                raise ValueError("this signal has no TaskConsole plot kind")
            wanted = str(inferred_kind)
        definition = task_console_panel_kind(wanted)
        selected_interval = self._panel_interval(
            self._default_interval_ms if interval_ms is None else interval_ms
        )
        self._panel_serial += 1
        panel_id = f"panel-{self._panel_serial}"
        state = PanelState(
            signal=signal_name,
            kind=definition.kind.value,
            cell_kind="",
            size=str(size).strip() or DEFAULTS.layout.default_preset,
            interval_ms=selected_interval,
            title=str(title).strip() or signal_name,
            semantic=dict(semantic or {}),
            display=dict(display or {}),
            fit=dict(fit or {}),
            overlay_signal=str(overlay_signal),
        )
        # Build the binding first so plot projection callbacks share its one
        # overlay resolver/revision stream from the initial frame onward.
        binding = PanelBinding(
            panel_id,
            state,
            None,
            None,
            parameter_surface=self._unbound_panel_parameters(state),
        )
        parameter_snapshot = initial
        if exact_value is not None and publication is not None:
            parameter_snapshot = (
                getattr(exact_value, "canonical_schema", None)
                or exact_value.snapshot.block.schema
            )
        projected = self._schema_projected_parameters(
            binding, parameter_snapshot, self._RESOLVING_REASON
        )
        if projected is not None:
            binding.parameter_surface = projected
        port = self._make_panel_port(binding)
        binding.port = port
        self.panels[panel_id] = binding

        # The id AND the title: a card is asked which panel it is by the board
        # and by the drop-order path, and asked what to caption itself by the
        # operator.  One argument for both meant every card in the real window
        # was captioned "Panel" and knew its own id as its title.
        self.view.add_panel(panel_id, binding.state.title)
        self.view.show_panel(panel_id, None)
        self.view.set_panel_selectors_enabled(panel_id, self._deriving)
        self._publish_panel_state(binding)
        self._refresh_console_projection()
        return binding

    @staticmethod
    def _publication_value(publication: object | None, signal: str) -> object | None:
        value = getattr(publication, "value", None)
        return value(str(signal)) if callable(value) else None

    def _make_panel_port(
        self,
        binding: PanelBinding,
        *,
        target: PanelState | None = None,
        notify_presented: bool = True,
        sync_history: bool = True,
    ) -> PlotPanelPort:
        """Wire one panel to the board through the single product path."""

        selected = binding.state if target is None else target
        port = PlotPanelPort(
            binding.panel_id,
            selected.signal,
            initial_target=selected,
            display_interval_ms=selected.interval_ms,
            companion_signals=lambda target: (
                (target.overlay_signal,) if target.overlay_signal else ()
            ),
            project_input=lambda value, pub, front, target: self._project_panel_input(
                binding, value, pub, front, state=target
            ),
            submit_projection=self.board.submit_projection,
            replace_host=lambda projected, value, pub, target: self._stage_panel_host(
                binding, projected, value, pub, state=target
            ),
            accept_host=lambda old: self._accept_panel_host(binding, old),
            retire_host=self._retire_plot_host,
            on_presented=(
                (lambda surface: self._panel_presented(binding, surface))
                if notify_presented
                else None
            ),
            present=lambda host, operation: self._present_panel_operation(
                binding, host, operation
            ),
            invalidate=lambda panel_id: self.board.invalidate_presentations(
                (panel_id,)
            ),
        )
        if sync_history:
            try:
                self._sync_panel_history(binding)
            except BaseException:
                port.close()
                raise
        return port

    def _presentation_snapshot(
        self,
        signal: str,
        value: object,
        publication: object,
    ) -> tuple[object, object]:
        """Resolve one Dataset together with its exact retained event record."""

        snapshot = getattr(value, "snapshot", None)
        if snapshot is None:
            raise TypeError("a panel signal value must carry an OwnedSnapshot")
        return self.session.signal_plane.current_dataset_view(
            signal,
            publication,
        )

    def _invalidate_signal_presentations(
        self,
        signal: str,
        *,
        semantic_schema: object | None = None,
    ) -> frozenset[str]:
        """Invalidate every surface whose signal-level Dataset mode changed."""

        selected = str(signal)
        invalidated: list[str] = []
        targets: dict[str, object] = {}
        for other in tuple(self.panels.values()):
            if other.state.signal != selected:
                continue
            if semantic_schema is not None:
                base = task_console_fitting_spec(
                    semantic_schema,
                    other.state.kind,
                    other.state.cell_kind,
                )
                if base is not None:
                    accepted = other.accepted_display
                    if accepted is not None:
                        previous = {
                            str(field.name)
                            for field in accepted.semantics.fields
                            if str(field.name) != "kind"
                        }
                        current = {
                            str(field.name)
                            for field in describe_semantics(
                                semantic_schema,
                                base,
                            ).fields
                            if str(field.name) != "kind"
                        }
                        semantic = {
                            str(name): value
                            for name, value in other.state.semantic.items()
                            if str(name) not in previous
                            or str(name) in current
                        }
                        if semantic != dict(other.state.semantic):
                            other.state = replace(
                                other.state,
                                semantic=semantic,
                            )
                projected = self._schema_projected_parameters(
                    other, semantic_schema, self._RESOLVING_REASON
                )
                if projected is not None:
                    other.parameter_surface = projected
            if other.port is not None:
                invalidated.append(other.panel_id)
                targets[other.panel_id] = other.state
            # A Frozen snapshot is an exact PlotInput representation.  An
            # event snapshot cannot honestly be re-labelled windowed/indexed
            # (or the reverse), even within the same run.
            if other.frozen_data is not None:
                other.refresh_requested = True
        if not self._closing:
            self.board.invalidate_presentations(
                invalidated,
                targets=targets,
            )
        return frozenset(invalidated)

    def _apply_history_transition(
        self,
        signal: str,
        transition: tuple[bool, bool, bool] | None,
    ) -> frozenset[str]:
        if transition is None:
            return frozenset()
        indexed, representation_changed, data_changed = transition
        if not (representation_changed or data_changed):
            return frozenset()
        semantic_schema = None
        if (
            representation_changed
            and not indexed
        ):
            publication = self.session.signal_plane.latest_publication(signal)
            value = None if publication is None else publication.value(signal)
            semantic_schema = (
                None
                if value is None
                else value.canonical_schema or value.snapshot.block.schema
            )
        return self._invalidate_signal_presentations(
            signal,
            semantic_schema=semantic_schema,
        )

    def _release_panel_history(
        self, binding: PanelBinding
    ) -> frozenset[str]:
        lease = binding.history_lease
        if lease is None:
            return frozenset()
        signal = lease.signal_name
        transition = lease.close()
        # Plane release is transactional.  Keep the handle until close
        # succeeds, otherwise an active demand becomes impossible to release.
        if binding.history_lease is lease:
            binding.history_lease = None
        return self._apply_history_transition(signal, transition)

    def _sync_panel_history(
        self,
        binding: PanelBinding,
        state: PanelState | None = None,
        *,
        schema: object | None = None,
    ) -> frozenset[str]:
        """Make Runtime retention exactly match this panel's current demand."""

        selected = binding.state if state is None else state
        signal = str(selected.signal)
        projection = self._panel_projection(
            binding,
            selected,
            schema=schema,
        )
        window = None
        if projection is not None:
            from zlc_plot.specs import parameter_schema_for

            spec, _semantic, display = projection
            effective = dict(
                parameter_schema_for(
                    spec,
                    style=DEFAULTS.style,
                ).initial_values(None)
            )
            effective.update(display)
            window = history_window_requirement(
                spec,
                effective,
            )
        if (
            window is not None
            and not self.session.signal_plane.supports_indexed_history(signal)
        ):
            window = None
        capable = bool(
            signal
            and window is not None
        )
        lease = binding.history_lease
        if not capable:
            return self._release_panel_history(binding)
        assert window is not None
        if lease is not None and lease.signal_name == signal:
            if lease.window != window:
                transition = lease.resize(window)
                return self._apply_history_transition(signal, transition)
            return frozenset()
        invalidated = set(self._release_panel_history(binding))
        binding.history_lease = self.session.signal_plane.acquire_indexed_history(
            signal,
            window,
        )
        invalidated.update(
            self._apply_history_transition(
                signal,
                binding.history_lease.acquisition_transition,
            )
        )
        return frozenset(invalidated)

    def _project_panel_input(
        self,
        binding: PanelBinding,
        value: object,
        publication: object,
        front: object = None,
        *,
        state: PanelState | None = None,
    ) -> object:
        """Compose the exact plot input for one immutable front.

        ``front`` is the plane's coherent freeze this panel is being drawn
        from, and it is the ONLY place an annotation is read: the port names
        the annotation among its front signals, so the plane freezes it at
        the shot of the picture.  Fetching it anywhere else -- the plane's
        latest, say -- is how rings from a later cycle land on this frame.
        """

        selected = binding.state if state is None else state
        snapshot, event_record = self._presentation_snapshot(
            selected.signal,
            value,
            publication,
        )
        event_records = ((publication, event_record),)
        # The SEMANTIC surface, not the outer kind: a FacetGrid of image cells
        # paints images, and the overlay is a fact of the image.  A grid over
        # curve cells has nowhere to put a ring.
        resolved = (
            self._panel_accepted_spec(binding)
            if state is None
            else None
        ) or self._panel_resolved_spec(
            binding, selected, subject=snapshot
        )
        if resolved is None or not paints_image_surface(resolved):
            return snapshot, event_records
        if selected.overlay_signal:
            if front is None:
                return snapshot, event_records
            overlay_result = self._image_point_overlay(
                front,
                publication,
                selected.overlay_signal,
                snapshot,
                binding.overlay_revision + 1,
            )
            if overlay_result is None:
                return snapshot, event_records
            overlay, overlay_publication, overlay_record = overlay_result
            event_records = (
                *event_records,
                (overlay_publication, overlay_record),
            )
        else:
            geometry = publication.run_record.get(
                IMAGE_POINT_OVERLAY_GEOMETRY_RECORD
            )
            if not isinstance(geometry, Mapping):
                return snapshot, event_records
            point_ids = tuple(str(value) for value in geometry["point_ids"])
            overlay = ImagePointOverlay(
                revision=binding.overlay_revision + 1,
                coordinates=geometry["coordinates_xy"],
                point_ids=point_ids,
                labels=tuple(str(value) for value in geometry["labels"]),
                static_statuses=tuple(
                    PointStatus.UNKNOWN for _ in point_ids
                ),
            )
        if overlay is None:
            return snapshot, event_records
        binding.overlay_revision += 1
        return ImageFrame(snapshot, overlay), event_records

    def _image_point_overlay(
        self,
        front: object,
        image_publication: object,
        overlay_signal: str,
        image_snapshot: object,
        revision: int,
    ) -> tuple[object, object, object] | None:
        """Ask the exact judgement publication for the layer to draw.

        Geometry and status both come out of the GIVEN FRONT.  Logic rows,
        bindings and live node objects are irrelevant: the retained
        publication is self-contained and must still draw, reload and save.
        """

        overlay_publication = front.publication(overlay_signal)
        if overlay_publication is None:
            return None
        if self.session.signal_plane.publication_roots(
            image_publication
        ) != self.session.signal_plane.publication_roots(overlay_publication):
            raise ValueError("image overlay publication is not from the image shot")
        pending = [overlay_publication]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if current is image_publication:
                break
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(
                self.session.signal_plane.direct_parent_publications(current)
            )
        else:
            raise ValueError(
                "image overlay publication does not descend from the image"
            )
        status = overlay_publication.value(overlay_signal)
        if status is None:
            return None
        snapshot = status.snapshot
        event_record = status.event_record
        geometry = overlay_publication.run_record.get(
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD
        )
        return (
            image_point_overlay_from_signal(
                geometry,
                snapshot,
                image_snapshot,
                revision=int(revision),
            ),
            overlay_publication,
            event_record,
        )

    def _panel_frozen_data(
        self,
        binding: PanelBinding,
        *,
        publication: object | None,
        plot_input: object,
        event_records: tuple[tuple[object, object], ...],
        target: PanelState,
        description: object,
    ) -> PanelFrozenData:
        overlay = (
            {"overlay_signal": target.overlay_signal}
            if isinstance(plot_input, ImageFrame) and target.overlay_signal
            else {}
        )
        return PanelFrozenData(
            publication,
            plot_input,
            target,
            description,
            capture_run_chain(
                self.session.signal_plane,
                event_records[-1][0] if event_records else publication,
                event_records=dict(reversed(event_records)),
                resolve_device_settings=(
                    self.session.resolve_device_setting_records
                ),
            ),
            overlay,
        )

    def _panel_presented(
        self,
        binding: PanelBinding,
        surface: object,
    ) -> None:
        """Track the exact live event separately from Panel Edit's frozen one."""

        publication = surface.publication
        plot_input = surface.plot_input
        description = surface.description
        describes_current_target = bool(
            description is not None
            and _same_panel_plot_target(surface.target, binding.state)
        )
        if describes_current_target:
            accepted_state = panel_state_from_description(
                binding.state,
                description,
            )
            state_changed = accepted_state != binding.state
            binding.state = accepted_state
            frozen_target = binding.state
        else:
            state_changed = False
            frozen_target = surface.target
        ui_changed = False
        if describes_current_target:
            controls = panel_surface_from_description(
                binding.state,
                description,
            )
            ui_changed = any(
                binding.parameter_surface.get(name) != controls.get(name)
                for name in controls
            )
            if ui_changed:
                binding.parameter_surface = controls
        interaction_changed = self._normalize_panel_interaction(binding)
        if binding.frozen_data is not None and not binding.refresh_requested:
            if state_changed or ui_changed or interaction_changed:
                self._publish_panel_state(binding)
            return
        binding.refresh_requested = False
        binding.frozen_data = self._panel_frozen_data(
            binding,
            publication=publication,
            plot_input=plot_input,
            event_records=surface.event_records,
            target=frozen_target,
            description=description,
        )
        self._publish_panel_state(binding)
        if binding.editor_open:
            try:
                self._replace_panel_editor_host(binding)
            except Exception as error:
                self._report(
                    f"cannot refresh {binding.state.title} plot editor: "
                    f"{_error_text(error)}",
                    severity="error",
                )

    # ------------------------------------------------------------ presentation
    #
    # Console panel widgets stage their fronts instead of auto-presenting them:
    # every render lands on screen through exactly one of these helpers, so a
    # causal group's members are presented together (the port's ``present``
    # inside one batch accept pass) and every non-batch render a host produces
    # -- configure, an armed fit, a mirrored selector -- still reaches pixels.

    def _panel_subject_schema(
        self,
        binding: PanelBinding,
        subject: object | None = None,
    ) -> object | None:
        """Canonical schema underneath one live/frozen interaction surface."""

        snapshot = self._shown_snapshot(binding) if subject is None else subject
        snapshot = getattr(snapshot, "snapshot", snapshot)
        schema = getattr(getattr(snapshot, "block", None), "schema", None)
        return self._panel_schema(binding) if schema is None else schema

    def _panel_projection(
        self,
        binding: PanelBinding,
        state: PanelState | None = None,
        *,
        subject: object | None = None,
        schema: object | None = None,
    ) -> tuple[object, dict[str, Any], dict[str, Any]] | None:
        """Resolve the authored state once through Plot's existing owner."""

        selected = binding.state if state is None else state
        if schema is None:
            schema = self._panel_subject_schema(binding, subject)
        if schema is None:
            return None
        spec = task_console_fitting_spec(
            schema, selected.kind, selected.cell_kind
        )
        if spec is None:
            return None
        return project_panel_state(schema, spec, selected)

    def _panel_resolved_spec(
        self,
        binding: PanelBinding,
        state: PanelState | None = None,
        *,
        subject: object | None = None,
        schema: object | None = None,
    ) -> object | None:
        try:
            projection = self._panel_projection(
                binding,
                state,
                subject=subject,
                schema=schema,
            )
        except SemanticVacancy:
            # An authored table with a vacant required role: legitimate
            # panel state that simply has no drawable specification.
            return None
        return None if projection is None else projection[0]

    @staticmethod
    def _panel_accepted_display(
        binding: PanelBinding,
        host: object | None = None,
    ) -> object | None:
        if host is not None and host is binding.editor_host:
            frozen = binding.frozen_data
            return None if frozen is None else frozen.description
        if host is None or host is binding.host:
            return binding.accepted_display
        return None

    @classmethod
    def _panel_accepted_spec(
        cls,
        binding: PanelBinding,
        host: object | None = None,
    ) -> object | None:
        display = cls._panel_accepted_display(binding, host)
        return None if display is None else display.spec

    @classmethod
    def _panel_accepted_subject(
        cls,
        binding: PanelBinding,
        host: object | None = None,
    ) -> object | None:
        display = cls._panel_accepted_display(binding, host)
        return None if display is None else display.selection_subject

    def _normalize_panel_interaction(self, binding: PanelBinding) -> bool:
        """Drop interaction state that cannot describe the current view."""

        description = binding.accepted_display
        if description is None:
            return False
        spec = description.spec
        subject = self._panel_accepted_subject(binding)
        state = binding.state
        schema = self._panel_subject_schema(binding)
        changes: dict[str, object] = {}
        if state.focused_cell is not None:
            from zlc_plot.semantics import axis_size
            from zlc_plot.specs import FacetGridPlot

            if (
                not isinstance(spec, FacetGridPlot)
                or schema is None
                or state.focused_cell >= axis_size(schema, spec.facet)
            ):
                changes["focused_cell"] = None
        if (
            state.classifier_thresholds
            and not accepts_classifier_thresholds(
                spec,
                description.display_state.values,
            )
        ):
            changes["classifier_thresholds"] = ()
        selection = panel_selection_from_document(state.selector)
        if (
            selection is not None
            and (
                subject is None
                or not panel_selection_matches_subject(selection, subject)
            )
        ):
            changes["selector"] = {}
        if changes:
            binding.state = replace(state, **changes)
            if "selector" in changes and binding.bridge is not None:
                binding.bridge.clear_selection()
        viewport_cleared = False
        if (
            binding.interaction_viewport is not None
            and binding.interaction_viewport[0]
            != self._panel_view_identity(binding)
        ):
            binding.interaction_viewport = None
            viewport_cleared = True
        return bool(changes) or viewport_cleared

    def _match_host_to_panel(
        self,
        binding: PanelBinding,
        host: object,
        *,
        state: PanelState | None = None,
        overlay: object = _UNCHANGED,
        display_updates: object = _UNCHANGED,
        restore_interaction: bool = False,
        interaction_input: object = _UNCHANGED,
    ) -> object:
        """Make one of this panel's hosts show what the panel says.

        A panel owns two hosts: the live card, and the Edit tab's copy of the
        same configuration over frozen data.  One PanelState reaches both;
        only their data differs, never appearance, fit or viewport.

        ``present`` is for a host whose widget STAGES its fronts: the card
        needs each resulting operation presented, while the Edit surface
        auto-presents its own.

        ``restore_interaction`` belongs only to a new/replacement/static host.
        A standing live host is the event authority for its committed Area,
        viewport, threshold and focus until the queued owner acknowledgement
        updates PanelState; replaying that briefly stale mirror during a Fit or
        display edit would erase the interaction that triggered the edit.
        """

        panel_state = binding.state if state is None else state
        interaction_subject = (
            None if interaction_input is _UNCHANGED else interaction_input
        )
        projection = self._panel_projection(
            binding,
            panel_state,
            subject=interaction_subject,
        )
        if projection is None:
            raise ValueError("panel target does not resolve on this Dataset")
        target_spec, _semantic, _display = projection
        interaction_spec = None
        classifier_thresholds: object = _UNCHANGED
        selectors: object = _UNCHANGED
        if restore_interaction:
            accepted_display = self._panel_accepted_display(binding, host)
            if accepted_display is None:
                accepted_display = binding.accepted_display
            interaction_spec = (
                None
                if accepted_display is None
                or accepted_display.spec != target_spec
                else accepted_display.spec
            )
            selection_subject = (
                None
                if accepted_display is None
                else accepted_display.selection_subject
            )
            if interaction_spec is not None:
                if accepts_classifier_thresholds(
                    interaction_spec, panel_state.display
                ):
                    classifier_thresholds = panel_state.classifier_thresholds
                selection = panel_selection_from_document(panel_state.selector)
                region_selectors = (
                    panel_plot_selectors(
                        selection,
                        facet_index=panel_state.focused_cell,
                    )
                    if selection is not None
                    and selection_subject is not None
                    and panel_selection_matches_subject(
                        selection, selection_subject
                    )
                    else ()
                )
                recorded = dict(panel_state.crosshair)
                if recorded:
                    from zlc_plot.selectors import (
                        CrosshairPoint,
                        SelectorState as _SelectorState,
                    )

                    crosshair_selectors: tuple = (
                        _SelectorState(
                            SelectorKind.CROSSHAIR,
                            CrosshairPoint(
                                float(recorded["x"]), float(recorded["y"])
                            ),
                        ),
                    )
                else:
                    crosshair_selectors = tuple(
                        selector
                        for selector in accepted_display.selectors
                        if selector.kind is SelectorKind.CROSSHAIR
                    )
                selectors = (
                    *region_selectors,
                    *crosshair_selectors,
                )
            else:
                # An unresolved host must not receive interaction state whose
                # coordinate vocabulary cannot yet be proved.
                selectors = ()
        selected_viewport = None
        if (
            restore_interaction
            and binding.interaction_viewport is not None
        ):
            measured_on, remembered = binding.interaction_viewport
            if measured_on == self._panel_view_identity(
                binding,
                state=panel_state,
                subject=(
                    None
                    if interaction_input is _UNCHANGED
                    else interaction_input
                ),
            ):
                selected_viewport = remembered
            else:
                # The data underneath moved: a range measured on the previous
                # one would paste its limits onto axes that no longer have
                # them, which is how a restarted producer came back framed by
                # the picture before it.
                binding.interaction_viewport = None
        _target_spec, semantic, display = projection
        configuration: dict[str, object] = {
            "semantic": semantic,
            "parameters": display,
            "size": panel_state.size,
            "fit": dict(panel_state.fit),
            "fit_live": True,
        }
        if restore_interaction:
            configuration.update(
                facet_focus=panel_state.focused_cell,
                viewport=selected_viewport,
            )
            if classifier_thresholds is not _UNCHANGED:
                configuration["classifier_thresholds"] = classifier_thresholds
            if selectors is not _UNCHANGED:
                configuration["selectors"] = selectors
        if overlay is not _UNCHANGED:
            configuration["image_overlay"] = overlay
        if display_updates is not _UNCHANGED:
            if not isinstance(display_updates, Mapping):
                raise TypeError("display_updates must be a mapping")
            # Keep the complete desired target so coalescing cannot lose an
            # earlier edit, and separately identify this transaction's authored
            # delta so Plot's transition owner can distinguish "switch away
            # from Fixed" from an explicit request to keep numeric bounds.
            # Treating the full PanelState as newly authored made Tight/Normal
            # retain old fixed bounds until remount.
            from zlc_plot.specs import parameter_schema_for

            configuration["parameter_updates"] = parameter_schema_for(
                target_spec,
                style=DEFAULTS.style,
            ).declared_subset(display_updates)
        pending = host.configure(**configuration)
        if host is binding.host:
            add = getattr(pending, "add_done_callback", None)
            if callable(add):
                add(lambda _future: self.board.wake.request_owner_wake())
        return pending

    def _remember_panel_view(self, binding: PanelBinding, **changes: Any) -> None:
        """Write down how this panel is being looked at.

        These belong to the panel's record -- they are saved with the board and
        restored with it -- but changing one is not a re-specification: the
        surface that produced the gesture already shows it, and the other one
        is mirrored beside this call.  So the record is replaced and published,
        without the configure path a Setting edit takes.
        """

        binding.state = replace(binding.state, **changes)
        self._publish_panel_state(binding)

    def _present_panel_operation(
        self,
        binding: PanelBinding,
        host: object,
        operation: object,
    ) -> bool:
        """Put one completed host operation on the panel's staged widget.

        THE moment a card changes what it shows.  Mounting the new host first
        and presenting later meant the card went blank for however long the
        replacement took to render -- a facet grid of twenty-five cells is
        seconds of nothing -- and any front that arrived in between met the
        previous host's widget.  Showing the host of the pixels that are
        landing keeps the old picture until the new one exists, and is a
        no-op once the card already shows it.
        """

        front = getattr(operation, "front", None)
        if front is None or host is None:
            return False
        # A front belongs to the host that painted it.  Renders queued by a
        # host this panel has since replaced still complete, and handing one
        # to the card would swap the card onto the new widget to show the old
        # host's pixels -- which the widget refuses, leaving a blank panel and
        # a red error.  A retired host's front is simply stale.
        identity = getattr(front, "identity", None)
        if getattr(identity, "host_id", None) != getattr(host, "host_id", None):
            return False
        present = getattr(self.view, "present_panel_front", None)
        if not callable(present):
            return False
        try:
            self.view.show_panel(binding.panel_id, host)
            accepted = bool(present(binding.panel_id, front))
        except Exception as error:
            # A front can lose its surface between completion and this present
            # (the operator reconfigured meanwhile).  The next render against
            # the new surface heals the pixels; the refusal is still recorded
            # so a panel that stops changing is never silent about why.
            binding.reported_error = error
            self._report(
                f"{binding.title}: {_error_text(error)}", severity="error"
            )
            accepted = False
        if accepted:
            return True
        previous = binding.accepted_surface
        if previous is not None and previous.host is not host:
            try:
                self.view.show_panel(binding.panel_id, previous.host)
                old_front = getattr(previous.host, "front", None)
                if old_front is not None:
                    if not bool(present(binding.panel_id, old_front)):
                        raise RuntimeError(
                            "the previous accepted panel front was refused"
                        )
            except Exception as restore_error:
                raise RuntimeError(
                    "candidate presentation failed and the previous panel "
                    "surface could not be restored"
                ) from restore_error
        return False

    def _present_when_done(self, binding: PanelBinding, operation: object) -> object:
        """Present a host operation's front once its worker completes.

        Completion callbacks fire on the plot worker; the present crosses to
        the GUI thread through the same interaction queue every other raster
        callback already uses, and is drained by the beat.
        """

        add = getattr(operation, "add_done_callback", None)
        if not callable(add):
            return operation
        host = binding.host

        def completed(future: object) -> None:
            try:
                result = future.result()
            except BaseException:
                # Cancelled or failed operations have no front; their errors
                # surface through the paths that already own them.
                return
            self._enqueue_panel_interaction(
                lambda: (
                    None
                    if host is None or binding.host is not host
                    else self._present_panel_operation(binding, host, result)
                )
            )

        add(completed)
        return operation

    def _track_panel_configuration(
        self,
        binding: PanelBinding,
        host: object,
        pending: object,
    ) -> None:
        """Accept a pure interaction description without touching authored state."""

        add = getattr(pending, "add_done_callback", None)
        if not callable(add):
            return

        def completed(future: object) -> None:
            def accept() -> None:
                live = host is binding.host
                current = (
                    binding.accepted_surface
                    if live
                    else binding.frozen_data
                    if host is binding.editor_host
                    else None
                )
                if current is None:
                    return
                try:
                    operation = future.result()
                    target = replace(
                        current.target,
                        selector=binding.state.selector,
                        classifier_thresholds=binding.state.classifier_thresholds,
                        focused_cell=binding.state.focused_cell,
                    )
                    if live:
                        if binding.port is None or binding.port.accept_configuration(
                            operation, target
                        ) is None:
                            raise RuntimeError(
                                "interactive plot front was not presented"
                            )
                    else:
                        binding.frozen_data = replace(
                            current,
                            target=target,
                            description=operation.value,
                        )
                    self._normalize_panel_interaction(binding)
                    self._publish_panel_state(binding)
                except Exception as error:
                    self._report(
                        f"cannot update {binding.state.title} interaction: "
                        f"{_error_text(error)}",
                        severity="error",
                    )

            self._enqueue_panel_interaction(accept)

        add(completed)

    @staticmethod
    def _cancel_panel_configuration(binding: PanelBinding) -> None:
        entry = binding.configuration
        binding.configuration = None
        if entry is None:
            return
        if entry[0] == "configure":
            entry[1].cancel()
            return
        if entry[0] == "retarget":
            _kind, _old_port, candidate_port, update, _target = entry
            update.future.cancel()
            candidate_port.close()

    def _stage_panel_host(
        self,
        binding: PanelBinding,
        plot_input: object,
        value: object,
        publication: object,
        *,
        state: PanelState,
    ) -> tuple[object, object]:
        """Fully configure a replacement without changing the mounted panel."""

        del value

        host = self._make_host(plot_input, state)
        try:
            operation = self._match_host_to_panel(
                binding,
                host,
                state=state,
                restore_interaction=True,
                interaction_input=plot_input,
            )
        except BaseException:
            self._retire_plot_host(host)
            raise
        return host, operation

    def _accept_panel_host(
        self,
        binding: PanelBinding,
        old_host: object,
    ) -> None:
        """Swap one staged generation without throwing out of cohort accept."""

        errors: list[BaseException] = []
        for resource in (binding.selections, binding.bridge):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
        binding.bridge = binding.selections = None
        if old_host is not None:
            try:
                self._retire_plot_host(old_host)
            except BaseException as error:
                errors.append(error)
        if errors:
            binding.reported_error = errors[0]
            self._report(
                f"{binding.title}: {_error_text(errors[0])}",
                severity="error",
            )

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
        """Tell one card which currently published signals it may show."""

        binding = self.panels[panel_id]
        publication = binding.display_publication
        if publication is None and binding.state.signal:
            publication = self.session.signal_plane.freeze().publication(
                binding.state.signal
            )
        self.view.set_panel_signal_choices(
            panel_id,
            self.signal_groups(),
            current=binding.state.signal,
            overlay_groups=self.overlay_signal_groups(
                binding.state.signal,
                publication,
            ),
            overlay_current=binding.state.overlay_signal,
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

    def _publication_families(self, publication: object) -> frozenset[tuple]:
        return frozenset(
            (root.stream_id, root.generation)
            for root in self.session.signal_plane.publication_roots(publication)
        )

    def overlay_signal_groups(
        self,
        signal: str,
        publication: object | None,
    ) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        """Site-status judgements describing the SAME SHOT as this image.

        Chosen by CONTRACT, so the console never learns a signal name.  Same
        shot, not same publication: occupancy reads the camera and publishes
        its own event, so requiring one publication hid the only pairing an
        operator ever wants -- a frame annotated with the judgement of that
        frame.
        """

        selected = str(signal).strip()
        if not selected or publication is None:
            return ()

        try:
            primary_families = self._publication_families(publication)
        except (LookupError, RuntimeError, ValueError):
            return ()

        contracts = dict(self._external_signal_contracts())
        for binding in self.logic.values():
            for output in self._logic_outputs(binding):
                contracts[stable_signal_key(binding.node_id, output.name)] = str(
                    output.contract_id
                )
        groups: dict[str, list[tuple[str, str]]] = {}
        for name, label, _state, producer, _derived in self.offered_signals(
            include_shown=True
        ):
            if contracts.get(name) != IMAGE_POINT_OVERLAY_CONTRACT:
                continue
            candidate = self.session.signal_plane.latest_publication(name)
            if candidate is None:
                continue
            try:
                candidate_families = self._publication_families(candidate)
            except (LookupError, RuntimeError, ValueError):
                continue
            if candidate_families == primary_families:
                groups.setdefault(producer or "signals", []).append((label, name))
        return tuple(
            (producer, tuple(leaves)) for producer, leaves in groups.items()
        )

    def _refresh_signal_choices(self) -> None:
        """Offer newly published signals on cards that are already open.

        Pushed only when the offer actually changes: a combo rebuilt on every
        beat is a combo that closes itself while an operator is reading it.
        """

        descriptions = {
            item.name: item for item in self.session.signal_plane.describe_signals()
        }
        projected = project_signals(self.session.signal_plane)
        panel_publishers = tuple(
            (
                panel_id,
                tuple(
                    (
                        row.name.rsplit("/", 1)[-1] or row.name,
                        format_signal_shape(descriptions[row.name].shape),
                        f"{row.state} · {row.name}",
                    )
                    for row in projected
                    if row.producer == panel_id
                ),
            )
            for panel_id in self.panels
        )
        if panel_publishers != self._shown_panel_publishers:
            self._shown_panel_publishers = panel_publishers
            self.view.set_panel_publishers(panel_publishers)

        groups = self.signal_groups()
        if groups == self._offered_groups:
            return
        self._offered_groups = groups
        front = self.session.signal_plane.freeze()
        for panel_id in self.view.panel_ids():
            binding = self.panels.get(panel_id)
            self.view.set_panel_signal_choices(
                panel_id,
                groups,
                current=binding.state.signal if binding is not None else "",
                overlay_groups=self.overlay_signal_groups(
                    binding.state.signal if binding is not None else "",
                    (
                        front.publication(binding.state.signal)
                        if binding is not None and binding.state.signal
                        else None
                    ),
                ),
                overlay_current=(
                    binding.state.overlay_signal if binding is not None else ""
                ),
            )
        for panel_id, binding in tuple(self.panels.items()):
            if (
                binding.port is None
                and binding.state.signal
                and front.value(binding.state.signal) is not None
            ):
                self.update_panel_state(panel_id, {"signal": binding.state.signal})
        for node_id in tuple(self.logic):
            self.refresh_logic_editor(node_id)

    def edit_panel(self, panel_id: str) -> bool:
        """Open or focus the panel's non-modal Edit projection.

        Opening it FREEZES the moment it is opened.  Edit shows one exact
        revision and holds it until Refresh, which is what makes a fit or a
        saved figure reproducible -- but the moment worth holding is the one
        the operator stepped aside to look at, not whenever this panel
        happened to be created.  A panel opened at the start of a scan and
        edited ten minutes later showed the scan's first point, which is not
        the picture the card beside it was showing, and no amount of looking
        at the Edit tab could say why.
        """

        binding = self.panels.get(panel_id)
        if binding is None:
            return False
        if binding.editor_host is None:
            # Focusing an already-open Edit keeps its frozen revision: that is
            # the one its fit and its Save Fig belong to.
            self.refresh_panel_snapshot(panel_id)
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
                    # A refused plot surface is a STATE of the editor, not a
                    # reason for it to close: the parameter form it carries
                    # is the one tool that can repair what the host refused
                    # (a stored y_min above y_max locked the operator out of
                    # the very field that would fix it).  The editor stays
                    # open, the refusal is reported, and every accepted
                    # state change offers the mount again.
                    self._report(
                        f"cannot mount {binding.state.title} plot editor: "
                        f"{_error_text(error)}",
                        severity="error",
                    )
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
        fit_override: object = _UNCHANGED
        allowed = {
            "signal",
            "kind",
            "cell_kind",
            "size",
            "interval_ms",
            "title",
            "semantic",
            "display",
            "fit",
            "overlay_signal",
        }
        unknown = tuple(name for name in changes if name not in allowed)
        if unknown:
            self._report(
                f"{panel_id}: unknown panel state field {unknown[0]!r}",
                severity="error",
            )
            return False
        current = binding.state
        science_fields = {"signal", "overlay_signal", "cell_kind", "semantic"}
        if science_fields.intersection(changes) and self._task_panel_science_blocked(
            binding,
            "changing signal, overlay, or data projection",
        ):
            return False
        if "kind" in changes and str(changes["kind"]) != current.kind:
            self._report(
                f"{panel_id}: plot kind is fixed; add another panel to use "
                f"{str(changes['kind']).replace('_', ' ')}",
                severity="warning",
            )
            return False
        if (
            "cell_kind" in changes
            and str(changes["cell_kind"]) != current.cell_kind
        ):
            try:
                task_console_panel_identity(current.kind, str(changes["cell_kind"]))
            except ValueError as error:
                self._report(f"{panel_id}: {error}", severity="warning")
                return False
        for name, check in (
            ("interval_ms", self._panel_interval),
            ("size", self._panel_size),
        ):
            if name in changes:
                try:
                    changes[name] = check(changes[name])
                except (TypeError, ValueError) as error:
                    self._report(f"{panel_id}: {_error_text(error)}", severity="error")
                    return False

        signal = str(changes.get("signal", current.signal)).strip()
        candidate_front = None
        candidate_value = None
        candidate_schema = None
        if signal:
            candidate_front = self.session.signal_plane.freeze()
            candidate_value = candidate_front.value(signal)
            if signal == current.signal and binding.accepted_surface is not None:
                accepted_input = binding.accepted_surface.plot_input
                accepted_snapshot = getattr(
                    accepted_input,
                    "snapshot",
                    accepted_input,
                )
                candidate_schema = accepted_snapshot.block.schema
            elif candidate_value is not None:
                candidate_schema = (
                    getattr(candidate_value, "canonical_schema", None)
                    or candidate_value.snapshot.block.schema
                )
        title = str(changes.get("title", current.title)).strip()
        if signal != current.signal and "title" not in changes:
            base_title = self._panel_kind_labels.get(current.kind, current.kind)
            suffix = current.title.removeprefix(f"{base_title} ")
            if current.title == current.signal or current.title == base_title or suffix.isdigit():
                title = signal
        if signal != current.signal and "overlay_signal" not in changes:
            changes["overlay_signal"] = ""
        title = (
            title
            or signal
            or self._panel_kind_labels.get(current.kind, current.kind.replace("_", " "))
        )
        merged: dict[str, Any] = {
            "signal": signal,
            # Validated above; without this line the patch was checked and
            # then silently dropped -- legal only while the old policy made
            # the current value the single legal one.
            "cell_kind": str(changes.get("cell_kind", current.cell_kind)),
            "size": str(changes.get("size", current.size)),
            "interval_ms": int(changes.get("interval_ms", current.interval_ms)),
            "title": title,
            "overlay_signal": str(
                changes.get("overlay_signal", current.overlay_signal)
            ),
        }
        if signal != current.signal:
            # A selector/focus/viewport names axes of the exact signal it was
            # drawn on.  A different signal must earn its own interaction
            # state; replaying the old axis ids onto it is not a restore.
            merged.update(
                selector={},
                classifier_thresholds=(),
                focused_cell=None,
            )
            binding.interaction_viewport = None
        projection_identity_changed = (
            signal != current.signal
            or str(changes.get("cell_kind", current.cell_kind))
            != current.cell_kind
        )
        for name in ("semantic", "display", "fit"):
            # Axis assignments and fit models belong to one exact
            # signal/cell vocabulary.  Carrying either into another and
            # silently filtering whatever no longer fits made a typo and an
            # old configuration indistinguishable.  Appearance remains
            # reusable across kinds; scientific projection and fitting start
            # from the new vocabulary's defaults.
            values = (
                {}
                if name in {"semantic", "fit"} and projection_identity_changed
                else dict(getattr(current, name))
            )
            if name in changes:
                try:
                    if name == "fit":
                        values, transient = fit_edit_targets(
                            values, dict(changes[name])
                        )
                        if transient is not None:
                            fit_override = transient
                    else:
                        values.update(dict(changes[name]))
                except (TypeError, ValueError) as error:
                    self._report(
                        f"{panel_id}: {_error_text(error)}", severity="error"
                    )
                    return False
            merged[name] = values
        fit_expression_update = fit_override is not _UNCHANGED
        if "semantic" in changes and candidate_schema is not None:
            base_state = replace(
                current,
                **{**merged, "semantic": {}},
            )
            base_spec = self._panel_resolved_spec(
                binding,
                base_state,
                schema=candidate_schema,
            )
            if base_spec is None:
                self._report(
                    f"{panel_id}: this signal has no compatible plot specification",
                    severity="error",
                )
                return False
            patch_state = replace(
                base_state,
                semantic=dict(changes["semantic"]),
                display={},
            )
            try:
                _resolved, semantic, _display = project_panel_state(
                    candidate_schema,
                    base_spec,
                    patch_state,
                )
            except SemanticVacancy as vacancy:
                # The operator's fates STICK.  A table whose required role
                # is vacant draws nothing -- that is the panel's state, not
                # a reason to bounce the edit back.
                semantic = {
                    **{
                        str(name): value
                        for name, value in dict(current.semantic).items()
                    },
                    **{
                        str(name): value
                        for name, value in dict(changes["semantic"]).items()
                    },
                }
                self._report(f"{panel_id}: {vacancy}", severity="info")
            except (KeyError, TypeError, ValueError) as error:
                self._report(
                    f"{panel_id}: {_error_text(error)}",
                    severity="error",
                )
                return False
            merged["semantic"] = semantic
        try:
            desired_overlay = str(merged["overlay_signal"])
            projection_candidate = replace(
                current,
                **{**merged, "overlay_signal": ""},
            )
            resolved_projection = (
                None
                if candidate_schema is None
                else self._panel_resolved_spec(
                    binding,
                    projection_candidate,
                    schema=candidate_schema,
                )
            )
            if (
                desired_overlay
                and resolved_projection is not None
                and not paints_image_surface(resolved_projection)
            ):
                # Overlay is a capability of the resolved image surface.  A
                # cell/signal transition that resolves to curves clears it in
                # the same state replacement, so no hidden companion can hold
                # the coherent front hostage.
                merged["overlay_signal"] = ""
            candidate = replace(current, **merged)
        except Exception as error:
            self._report(f"{panel_id}: {_error_text(error)}", severity="error")
            return False
        if "display" in changes:
            # What no host could ever accept must never be STORED: an
            # inverted limit pair used to pass through here, fail the host
            # at its next start, and lock the operator out of every surface
            # that could repair it.  The appearance bag keeps other kinds'
            # vocabulary; the contract judges this kind's own subset, and
            # an INCOMPLETE state still passes -- fixed limits materialize
            # on the next configure.
            try:
                resolved = (
                    None
                    if candidate_schema is None
                    else self._panel_resolved_spec(
                        binding,
                        candidate,
                        schema=candidate_schema,
                    )
                )
                resolved_cell = (
                    semantic_spec(resolved).kind.value
                    if resolved is not None
                    and candidate.kind == PlotKind.FACET_GRID.value
                    else candidate.cell_kind or None
                )
                validate_authored_display(
                    candidate.kind,
                    merged["display"],
                    style=DEFAULTS.style,
                    facet_cell_kind=resolved_cell,
                )
            except (TypeError, ValueError, KeyError) as error:
                self._report(
                    f"{panel_id}: {_error_text(error)}", severity="error"
                )
                return False
        plot_changed = any(
            getattr(candidate, name) != getattr(current, name)
            for name in (
                "size",
                "semantic",
                "display",
                "fit",
                "overlay_signal",
            )
        ) or fit_expression_update
        if (
            candidate.overlay_signal
            and candidate.overlay_signal != current.overlay_signal
        ):
            front = candidate_front or self.session.signal_plane.freeze()
            publication = (
                binding.display_publication
                or front.publication(candidate.signal)
            )
            offered = {
                name
                for _producer, leaves in self.overlay_signal_groups(
                    candidate.signal,
                    publication,
                )
                for _label, name in leaves
            }
            if candidate.overlay_signal not in offered:
                self._report(
                    f"{panel_id}: this console publishes no "
                    f"{candidate.overlay_signal!r} overlay that describes "
                    f"{candidate.signal!r}",
                    severity="error",
                )
                return False
        needs_mount = (
            bool(candidate.signal)
            and (
                candidate.signal != current.signal
                # The cell kind is SPEC-level identity: the plot host draws
                # the spec it was built with and only reconfigures display
                # parameters after that.  A changed cell must rebuild the
                # host through the same path a changed signal takes --
                # accepting the state and keeping the old picture is a
                # control that looks live and does nothing.
                or candidate.cell_kind != current.cell_kind
                or binding.port is None
                or binding.host is None
                or (
                    binding.configuration is not None
                    and binding.configuration[0] == "retarget"
                )
                # A host that failed STARTUP is permanently unusable, and
                # reconfiguring it re-raises the same reason forever.  The
                # accepted state change -- the repair -- rebuilds instead.
                or getattr(binding.host, "startup_failure", None) is not None
            )
        )
        if (
            candidate == current
            and not needs_mount
            and not fit_expression_update
        ):
            return False
        if not candidate.signal:
            if binding.host is not None or binding.port is not None:
                self._release_panel_editor(binding)
                self._release_panel(binding)
                self.view.show_panel(panel_id, None)
            binding.state = candidate
            binding.parameter_surface = self._unbound_panel_parameters(candidate)
            binding.frozen_data = None
            self._publish_panel_state(binding)
            self._refresh_console_projection()
            return True

        if needs_mount:
            front = candidate_front
            value = candidate_value
            if value is None:
                if candidate.signal != current.signal and binding.port is not None:
                    self._report(
                        f"{candidate.signal} has not published yet",
                        severity="warning",
                    )
                    return False
                binding.state = candidate
                binding.parameter_surface = self._unbound_panel_parameters(candidate)
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                self._report(
                    f"{candidate.signal} has not published yet; {panel_id} remains ready",
                    severity="warning",
                )
                return True
            fitting = (
                self._fitting_cell_kind(
                    value,
                    candidate.kind,
                    candidate.cell_kind,
                )
                if candidate.kind
                else candidate.cell_kind
            )
            # An empty cell kind stays empty: the data decides this time and
            # keeps deciding on every later retarget.  Writing the derived
            # value back would turn one dataset's answer into an authored
            # choice the next dataset gets refused against.
            if candidate.kind and fitting is not None and fitting != candidate.cell_kind:
                self._report(
                    f"{panel_id} draws {fitting} cells for {candidate.signal}",
                    severity="task",
                )
            if candidate.kind and fitting is None:
                if candidate.signal != current.signal and binding.port is not None:
                    self._report(
                        f"{candidate.signal} cannot be drawn as a "
                        f"{candidate.kind.replace('_', ' ')}",
                        severity="warning",
                    )
                    return False
                binding.state = candidate
                binding.parameter_surface = self._unbound_panel_parameters(candidate)
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                self._report(
                    f"{candidate.signal} cannot be drawn as a "
                    f"{candidate.kind.replace('_', ' ')}; the panel remains ready",
                    severity="warning",
                )
                return True

            parameter_schema = (
                getattr(value, "canonical_schema", None)
                or value.snapshot.block.schema
            )
            publication = front.publication(candidate.signal)
            if publication is None:
                self._report(
                    f"{candidate.signal} has no exact publication",
                    severity="error",
                )
                return False
            self._cancel_panel_configuration(binding)
            binding.state = candidate
            binding.parameter_surface = (
                self._schema_projected_parameters(
                    binding,
                    parameter_schema,
                    self._RESOLVING_REASON,
                )
                or self._unbound_panel_parameters(candidate)
            )
            candidate_port = self._make_panel_port(
                binding,
                target=candidate,
                notify_presented=False,
                sync_history=False,
            )
            try:
                update = candidate_port.prepare(value, publication, front)
                if update is None:
                    raise RuntimeError("candidate panel produced no surface update")
            except BaseException as error:
                candidate_port.close()
                self._report(
                    f"{panel_id}: {_error_text(error)}",
                    severity="error",
                )
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                return True
            old_port = binding.port
            binding.configuration = (
                "retarget",
                old_port,
                candidate_port,
                update,
                candidate,
            )
            update.future.add_done_callback(
                lambda _future: self.board.wake.request_owner_wake()
            )
            self._report(
                f"{panel_id} is preparing {candidate.signal}", severity="task"
            )
        else:
            if candidate.interval_ms != current.interval_ms and binding.port is not None:
                binding.port.set_display_interval(candidate.interval_ms)
            binding.state = candidate
            if binding.host is None or binding.port is None:
                schema = self._panel_schema(binding)
                binding.parameter_surface = (
                    (
                        None
                        if schema is None
                        else self._schema_projected_parameters(
                            binding, schema, self._RESOLVING_REASON
                        )
                    )
                    or self._unbound_panel_parameters(candidate)
                )
                self._remount_panel_editor(binding)
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                return True
            if plot_changed:
                # Both hosts already own the immutable Dataset they display. A
                # history-window edit is a Plot projection over those bytes;
                # Runtime's lease controls only what future publications retain.
                # Asking Runtime to re-materialize ``display_publication`` here
                # raced generation replacement and tried to resurrect retired
                # ROI/Fit publications.
                live_overlay: object = _UNCHANGED
                if (
                    self._paints_image_surfaces(binding, candidate)
                    and candidate.overlay_signal != current.overlay_signal
                ):
                    live_overlay = None
                self._cancel_panel_configuration(binding)
                pending = self._match_host_to_panel(
                    binding,
                    binding.host,
                    state=(
                        candidate
                        if fit_override is _UNCHANGED
                        else replace(candidate, fit=fit_override)
                    ),
                    overlay=live_overlay,
                    display_updates=changes.get("display", _UNCHANGED),
                    restore_interaction="semantic" in changes,
                )
                binding.configuration = (
                    "configure",
                    pending,
                    True,
                    candidate,
                )
            else:
                self._remount_panel_editor(binding)

        self._publish_panel_state(binding)
        self._refresh_console_projection()
        return True

    def _parameter_surface(
        self,
        controls: Sequence[object],
        state: PanelState,
        *,
        semantic: Sequence[Mapping[str, object]] = (),
        fit: Sequence[Mapping[str, object]] = (),
        semantic_unavailable: str = "",
        display_unavailable: str = "",
        fit_unavailable: str = "",
        fit_outputs: Sequence[tuple[str, str]] = (),
        semantic_provisional: bool = False,
    ) -> Mapping[str, object]:
        """Project one plot-owned display declaration for every panel view."""

        display_entries: list[dict[str, object]] = []
        for control in controls:
            entry = control_document(control)
            display_entries.append(entry)
        return {
            "semantic": tuple(semantic),
            "display": tuple(display_entries),
            "fit": tuple(fit),
            "semantic_unavailable": str(semantic_unavailable),
            "display_unavailable": str(display_unavailable),
            "fit_unavailable": str(fit_unavailable),
            "fit_outputs": tuple((str(name), str(label)) for name, label in fit_outputs),
            # Schema-projected, not yet host-described: the fate rows are
            # real (they come from schema+spec alone) but choices are not
            # feasibility-filtered and fit models are absent, so the host's
            # description still replaces this surface when it arrives.
            "semantic_provisional": bool(semantic_provisional),
        }

    def _unbound_panel_parameters(self, state: PanelState) -> Mapping[str, object]:
        """Describe a fixed kind before a compatible dataset is connected."""

        values = dict(state.display)
        data_reason = "Choose a compatible signal to resolve dataset-dependent choices."
        if state.kind == PlotKind.FACET_GRID.value and not state.cell_kind:
            # An empty cell kind means the DATA decides; until a signal binds
            # there is no cell, so there is no display contract to show yet.
            # That is this panel's authored state, not a configuration error.
            return self._parameter_surface(
                (),
                state,
                semantic_unavailable=data_reason,
                display_unavailable=data_reason,
                fit_unavailable=data_reason,
            )
        try:
            controls = parameter_controls_for_kind(
                state.kind,
                values,
                facet_cell_kind=state.cell_kind or None,
            )
        except (TypeError, ValueError, KeyError) as error:
            controls = ()
            display_unavailable = _error_text(error)
        else:
            display_unavailable = ""
        return self._parameter_surface(
            controls,
            state,
            semantic_unavailable=data_reason,
            display_unavailable=display_unavailable,
            fit_unavailable=data_reason,
        )

    def _schema_projected_parameters(
        self,
        binding: PanelBinding,
        snapshot: object,
        reason: str,
    ) -> Mapping[str, object] | None:
        """The semantic contract projected from schema + spec alone.

        THE light path: describe_semantics needs no render, so the fate
        rows appear the moment a compatible snapshot is in hand -- at
        connect, at accept, and on a panel whose HOST could not mount.  A
        refused projection (the facet cell cap, say) is a STATE of the
        panel, not a reason for it to have no form: the semantic choices --
        the very fates that fix the refusal -- come from the schema and the
        spec alone, so a dead host cannot take them away.  Display controls
        come from the kind vocabulary; only fit truly needs a live host,
        and the host's own description still replaces this surface when it
        arrives (feasibility-filtered choices, exact values, fit models).
        """

        from zlc_plot.semantics import describe_semantics
        from zlc_plot.ui import parameter_controls_for_kind

        from zlc_data import DatasetSchema

        schema = (
            snapshot
            if isinstance(snapshot, DatasetSchema)
            else getattr(getattr(snapshot, "block", None), "schema", None)
        )
        if schema is None:
            return None
        state = binding.state
        spec = task_console_fitting_spec(
            schema, state.kind, state.cell_kind
        )
        if spec is None:
            return None
        semantic_reason = ""
        authored: Mapping[str, object] = {}
        try:
            resolved, _semantic, _display = project_panel_state(
                schema, spec, state
            )
            description = describe_semantics(schema, resolved)
        except Exception as error:
            # The form is the repair surface.  Keep the schema's exact field
            # domain visible, but keep the refusal loud; never pretend the
            # default spec was the accepted view.
            resolved = spec
            try:
                description = describe_semantics(schema, resolved)
            except Exception:
                return None
            reason = _error_text(error)
            if isinstance(error, SemanticVacancy):
                # The operator's own table, not the default's: the fates
                # they chose stay on the form, with the vacancy as the
                # reason nothing draws.
                semantic_reason = _error_text(error)
                authored = dict(state.semantic)
        actual_cell_kind = (
            semantic_spec(resolved).kind.value
            if state.kind == PlotKind.FACET_GRID.value
            else state.cell_kind or None
        )
        try:
            controls = parameter_controls_for_kind(
                state.kind,
                dict(state.display),
                facet_cell_kind=actual_cell_kind,
            )
        except (TypeError, ValueError, KeyError):
            controls = ()
        entries = list(semantic_entries(description))
        if authored:
            shown = []
            for entry in entries:
                name = str(entry.get("name", ""))
                if name in authored:
                    entry = {**entry, "value": authored[name]}
                shown.append(entry)
            entries = shown
        return self._parameter_surface(
            controls,
            state,
            semantic=tuple(entries),
            semantic_unavailable=semantic_reason,
            fit_unavailable=str(reason),
            semantic_provisional=True,
        )

    #: Why fit is absent on a surface projected before the host settles.
    _RESOLVING_REASON = "Fit models resolve when the plot surface mounts."

    def _panel_schema(self, binding: PanelBinding) -> object | None:
        """The canonical schema a Panel edits, even before it can draw.

        Exact producers publish one event chunk plus a canonical run schema.
        Settings and the title need only the latter and must not materialize
        the full Dataset on the GUI thread merely to list axes.  A Monitor has
        no canonical extent, so its latest event schema remains the answer.
        """

        def value_schema(value: object) -> object | None:
            canonical = getattr(value, "canonical_schema", None)
            snapshot = getattr(value, "snapshot", None)
            return (
                canonical
                if canonical is not None
                else getattr(getattr(snapshot, "block", None), "schema", None)
            )

        accepted = binding.accepted_surface
        accepted_target = None if accepted is None else accepted.target
        shown = (
            None
            if accepted is None
            or getattr(accepted_target, "signal", None) != binding.state.signal
            else accepted.plot_input
        )
        shown = getattr(shown, "snapshot", shown)
        schema = getattr(getattr(shown, "block", None), "schema", None)
        if schema is not None:
            return schema
        frozen = binding.frozen_data
        if frozen is not None and frozen.signal == binding.state.signal:
            return getattr(frozen.snapshot.block, "schema", None)
        publication = (
            binding.display_publication
            if getattr(accepted_target, "signal", None) == binding.state.signal
            else None
        )
        if publication is not None:
            schema = value_schema(publication.value(binding.state.signal))
            if schema is not None:
                return schema
        if binding.state.signal:
            value = self.session.signal_plane.freeze().value(binding.state.signal)
            if value is not None:
                return value_schema(value)
        return None

    def _degrade_panel_surface(
        self, binding: PanelBinding, error: BaseException
    ) -> None:
        """Keep the semantic form alive when a host fails to mount."""

        schema = self._panel_schema(binding)
        if schema is None:
            return
        surface = self._schema_projected_parameters(
            binding, schema, _error_text(error)
        )
        if surface is None:
            return
        binding.parameter_surface = surface
        try:
            self._publish_panel_state(binding)
        except Exception:
            pass
        self.refresh_panel_editor(binding.panel_id)

    def _settle_panel_hosts(self) -> None:
        """Project metadata and selectors only after each initial render finished."""

        for binding in tuple(self.panels.values()):
            host = binding.host
            configuration_entry = binding.configuration
            if (
                configuration_entry is not None
                and configuration_entry[0] == "retarget"
            ):
                (
                    _kind,
                    old_port,
                    candidate_port,
                    update,
                    target,
                ) = configuration_entry
                if update.future.done():
                    if binding.configuration is configuration_entry:
                        binding.configuration = None
                    swapped = False
                    try:
                        operation = update.future.result()
                        normalized_target = panel_state_from_description(
                            target,
                            operation.value,
                        )
                        if binding.state not in (target, normalized_target):
                            raise RuntimeError(
                                "panel target changed while replacement rendered"
                            )
                        if not candidate_port.can_accept(update, operation):
                            raise RuntimeError(
                                "candidate panel surface is no longer current"
                            )
                        candidate_port.accept(update, operation)
                        accepted = candidate_port.accepted_surface()
                        if accepted is None:
                            raise RuntimeError(
                                "candidate panel front was not presented"
                            )
                        old_surface = (
                            None if old_port is None else old_port.accepted_surface()
                        )
                        old_host = None if old_surface is None else old_surface.host
                        binding.port = candidate_port
                        swapped = True
                        if old_port is not None:
                            old_port.close()
                        if old_host is not None and old_host is not accepted.host:
                            self._retire_plot_host(old_host)
                        host = binding.host
                    except Exception as error:
                        if not swapped:
                            candidate_port.reject(update, error)
                            candidate_port.close()
                        else:
                            self._panel_presented(binding, accepted)
                        binding.reported_error = error
                        self._report(
                            f"{binding.panel_id}: {_error_text(error)}",
                            severity="error",
                        )
                        self._degrade_panel_surface(binding, error)
                    else:
                        try:
                            self._sync_panel_history(binding)
                        except Exception as error:
                            binding.reported_error = error
                            self._report(
                                f"{binding.panel_id}: {_error_text(error)}",
                                severity="error",
                            )
                        self._panel_presented(binding, accepted)
                configuration_entry = binding.configuration
            if (
                configuration_entry is not None
                and configuration_entry[0] == "configure"
            ):
                (
                    _kind,
                    pending,
                    normalize_state,
                    configuration_target,
                ) = configuration_entry
            else:
                pending = None
                normalize_state = False
                configuration_target = None
            if pending is not None and pending.done():
                binding.configuration = None
                if not pending.cancelled():
                    try:
                        operation = pending.result()
                        accepted = (
                            None
                            if binding.port is None
                            else binding.port.accept_configuration(
                                operation,
                                configuration_target,
                            )
                        )
                        if accepted is None:
                            raise RuntimeError(
                                "configured plot front was not presented"
                            )
                        description = accepted.description
                    except Exception as error:
                        if binding.reported_error is not error:
                            binding.reported_error = error
                            self._report(
                                f"{binding.panel_id}: {_error_text(error)}",
                                severity="error",
                            )
                            self._degrade_panel_surface(binding, error)
                    else:
                        assert description is not None
                        if normalize_state:
                            surface = panel_surface_from_description(
                                binding.state,
                                description,
                            )
                            binding.state = panel_state_from_description(
                                binding.state,
                                description,
                            )
                            binding.parameter_surface = surface
                            self._sync_panel_history(binding)
                        if binding.interaction_viewport is not None:
                            binding.interaction_viewport = (
                                None
                                if description.viewport is None
                                else (
                                    self._panel_view_identity(binding),
                                    description.viewport,
                                )
                            )
                        if normalize_state:
                            self._normalize_panel_interaction(binding)
                        self._publish_panel_state(binding)
                        frozen = binding.frozen_data
                        if (
                            normalize_state
                            and frozen is not None
                            and frozen.signal == binding.state.signal
                            and _run_of(frozen.publication)
                            == _run_of(binding.display_publication)
                        ):
                            frozen_input = frozen.plot_input
                            frozen_overlay = frozen.overlay
                            if (
                                frozen.target.overlay_signal
                                != binding.state.overlay_signal
                            ):
                                frozen_input = frozen.snapshot
                                frozen_overlay = {}
                            candidate_frozen = replace(
                                frozen,
                                plot_input=frozen_input,
                                target=binding.state,
                                description=description,
                                overlay=frozen_overlay,
                            )
                            if binding.editor_open:
                                try:
                                    self._replace_panel_editor_host(
                                        binding,
                                        frozen=candidate_frozen,
                                    )
                                except Exception as error:
                                    self._report(
                                        f"cannot update {binding.state.title} "
                                        f"plot editor: {_error_text(error)}",
                                        severity="error",
                                    )
                            else:
                                binding.frozen_data = candidate_frozen
            editor_entry = binding.editor_configuration
            if editor_entry is not None:
                (
                    editor_host,
                    editor_pending,
                    normalize_editor_state,
                    editor_target,
                    editor_frozen,
                ) = editor_entry
            else:
                editor_host = editor_pending = None
                normalize_editor_state = False
                editor_target = None
                editor_frozen = None
            if editor_pending is not None and editor_pending.done():
                if binding.editor_configuration is editor_entry:
                    binding.editor_configuration = None
                try:
                    if editor_pending.cancelled():
                        raise RuntimeError("plot editor configuration was cancelled")
                    editor_operation = editor_pending.result()
                    description = editor_operation.value
                    normalized_target = (
                        panel_state_from_description(binding.state, description)
                        if normalize_editor_state
                        else editor_target
                    )
                    current_frozen = binding.frozen_data
                    mount = getattr(self.view, "show_panel_editor", None)
                    if (
                        not binding.editor_open
                        or editor_frozen is None
                        or current_frozen is None
                        or not _same_panel_plot_target(
                            binding.state,
                            editor_target,
                        )
                        or current_frozen.publication is not editor_frozen.publication
                        or current_frozen.snapshot.ref != editor_frozen.snapshot.ref
                        or not callable(mount)
                    ):
                        raise RuntimeError(
                            "plot editor target is no longer current"
                        )
                    accepted_frozen = replace(
                        editor_frozen,
                        target=normalized_target,
                        description=description,
                    )
                    selections = self._subscribe_editor_gestures(
                        binding,
                        editor_host,
                        accepted_frozen,
                    )
                    old_host = binding.editor_host
                    old_selections = binding.editor_selections
                    try:
                        mount(binding.panel_id, editor_host)
                    except BaseException:
                        selections.close()
                        raise
                    binding.editor_host = editor_host
                    binding.editor_selections = selections
                    binding.frozen_data = accepted_frozen
                    if normalize_editor_state:
                        binding.state = normalized_target
                        binding.parameter_surface = panel_surface_from_description(
                            normalized_target,
                            description,
                        )
                        self._normalize_panel_interaction(binding)
                        self._publish_panel_state(binding)
                    if old_selections is not None:
                        old_selections.close()
                    if old_host is not None:
                        self._retire_plot_host(old_host)
                    self.refresh_panel_editor(binding.panel_id)
                except Exception as error:
                    if editor_host is not binding.editor_host:
                        self._retire_plot_host(editor_host)
                    self._report(
                        f"cannot update {binding.state.title} plot editor: "
                        f"{_error_text(error)}",
                        severity="error",
                    )
            if host is not None:
                self._apply_deriving(binding)


    def _direct_producer_node_id(self, signal: str) -> str | None:
        for binding in self.logic.values():
            if any(
                stable_signal_key(binding.node_id, output.name) == str(signal)
                for output in self._logic_outputs(binding)
            ):
                return binding.node_id
        return None

    @staticmethod
    def _logic_outputs(binding: LogicBinding) -> tuple[object, ...]:
        """Outputs of the active run, or of the exact stopped draft."""

        if binding.host is not None:
            return tuple(binding.host.dataset_output_declarations)
        return tuple(binding.descriptor.outputs)

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
            "overlay_signal_options": self.overlay_signal_groups(
                binding.state.signal,
                (
                    frozen.publication
                    if frozen is not None
                    else binding.display_publication
                ),
            ),
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
            "save_directory": str(self.session.day_folder()),
        }

    def refresh_panel_editor(self, panel_id: str) -> bool:
        projection = self.panel_editor_projection(panel_id)
        if projection is None:
            return False
        update = getattr(self.view, "update_panel_editor", None)
        if callable(update):
            update(str(panel_id), projection)
        return True

    def _panel_publisher_fields(
        self,
        binding: PanelBinding,
    ) -> tuple[tuple[str, str], ...]:
        fields = list(
            panel_selection_output_catalog(
                self._panel_accepted_subject(binding)
            )
        )
        description = binding.accepted_display
        if description is not None:
            fields.extend(
                fit_output_fields(description.fit, description.fit_models)
            )
        return tuple(fields)

    def panel_publisher_editor_projection(
        self,
        panel_id: str,
    ) -> dict[str, Any] | None:
        binding = self.panels.get(str(panel_id))
        if binding is None:
            return None
        fields = self._panel_publisher_fields(binding)
        values = {
            name: binding.state.published_outputs.get(name, True)
            for name, _label in fields
        }
        return {
            "node_id": binding.panel_id,
            "api_name": f"{binding.state.title} outputs",
            "kind": "panel publisher",
            "form_spec": FormSpec(
                tuple(
                    FormFieldProps(
                        key=name,
                        kind="bool",
                        label=label,
                        default=True,
                    )
                    for name, label in fields
                )
            ),
            "form_values": values,
            "source_required": False,
            "device_options": {},
            "device_keys": {},
            "preview_offered": False,
            "auto_preview": False,
            "running": False,
            "pending": False,
            "can_start": False,
            "can_stop": False,
            "science_locked": self._task_science_locked(binding),
            "issues": (),
            "error": "",
            "status": "",
        }

    def edit_panel_publisher(self, panel_id: str) -> bool:
        projection = self.panel_publisher_editor_projection(panel_id)
        if projection is None:
            return False
        opened = getattr(self.view, "open_panel_publisher_editor", None)
        focused = getattr(self.view, "focus_panel_publisher_editor", None)
        if callable(opened):
            opened(str(panel_id), projection)
        if callable(focused):
            focused(str(panel_id))
        return callable(opened) or callable(focused)

    def refresh_panel_publisher_editor(self, panel_id: str) -> bool:
        projection = self.panel_publisher_editor_projection(panel_id)
        update = getattr(self.view, "update_panel_publisher_editor", None)
        if projection is None or not callable(update):
            return False
        return bool(update(str(panel_id), projection))

    def _panel_publisher_draft_changed(
        self,
        panel_id: str,
        patch: Mapping[str, Any],
    ) -> None:
        values = patch.get("values", {})
        if isinstance(values, Mapping):
            self.update_panel_published_outputs(str(panel_id), values)

    def update_panel_published_outputs(
        self,
        panel_id: str,
        values: Mapping[str, Any],
    ) -> bool:
        """Replace publisher policy without reconfiguring the plot host."""

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return False
        if self._task_panel_science_blocked(
            binding,
            "changing derived panel outputs",
        ):
            return False
        published = dict(binding.state.published_outputs)
        published.update({str(name): bool(enabled) for name, enabled in values.items()})
        candidate = replace(binding.state, published_outputs=published)
        if candidate == binding.state:
            return False
        binding.state = candidate
        if binding.bridge is not None:
            binding.bridge.configure_outputs(candidate.published_outputs)
        self._publish_panel_state(binding)
        self._refresh_signal_choices()
        self._refresh_console_projection()
        return True

    def _remount_panel_editor(self, binding: PanelBinding) -> None:
        """Offer Edit's plot surface to an editor open without one.

        An Edit whose host refused to start stays OPEN -- its form is the
        tool that repairs the state the host refused -- so every accepted
        state change offers the mount again, and a failure is a report,
        never a closed window.
        """

        if not binding.editor_open:
            return
        if (
            binding.editor_host is not None
            and getattr(binding.editor_host, "startup_failure", None) is None
        ):
            return
        frozen = binding.frozen_data
        if (
            frozen is None
            or frozen.signal != binding.state.signal
            or (
                binding.display_publication is not None
                and _run_of(frozen.publication)
                != _run_of(binding.display_publication)
            )
        ):
            return
        try:
            self._replace_panel_editor_host(binding)
        except Exception as error:
            self._report(
                f"cannot mount {binding.state.title} plot editor: "
                f"{_error_text(error)}",
                severity="error",
            )

    def _replace_panel_editor_host(
        self,
        binding: PanelBinding,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> object:
        """Stage Edit completely; its old accepted surface stays visible."""

        frozen = binding.frozen_data if frozen is None else frozen
        if frozen is None:
            raise RuntimeError(f"{binding.panel_id} has no frozen plot input")
        plot_input = frozen.plot_input
        host = self._make_host(plot_input, binding.state)

        if not callable(getattr(self.view, "show_panel_editor", None)):
            self._retire_plot_host(host)
            raise RuntimeError("this console cannot mount a Panel Edit plot surface")
        previous = binding.editor_configuration
        if previous is not None:
            previous_host, previous_pending, _normalize, _target, _frozen = previous
            previous_pending.cancel()
            if previous_host is not binding.editor_host:
                self._retire_plot_host(previous_host)
        try:
            pending = self._match_host_to_panel(
                binding,
                host,
                restore_interaction=True,
                interaction_input=plot_input,
            )
        except BaseException:
            self._retire_plot_host(host)
            raise
        binding.editor_configuration = (
            host,
            pending,
            True,
            binding.state,
            frozen,
        )
        return host

    def _refresh_panel_editor_selection(self, binding: PanelBinding) -> None:
        """Rebind one unchanged editor host to a replaced frozen record."""

        host = binding.editor_host
        frozen = binding.frozen_data
        if host is None or frozen is None:
            return
        selections = self._subscribe_editor_gestures(binding, host, frozen)
        previous = binding.editor_selections
        binding.editor_selections = selections
        if previous is not None:
            previous.close()

    def _subscribe_editor_gestures(
        self,
        binding: PanelBinding,
        host: object,
        frozen: PanelFrozenData,
    ) -> object:
        """Listen to everything the operator can do on Edit's frozen surface.

        Mounting the surface and re-binding an unchanged one to a newer freeze
        both need exactly this, and they were written out twice: the second
        copy silently lacked the threshold channel, so a level set before a
        Refresh reached the panel and one set after it did not.
        """

        panel_id = binding.panel_id

        source = PlotSelectionSource(
            host,
            on_threshold=lambda event: self._enqueue_panel_threshold(
                panel_id, host, event, frozen=frozen
            ),
            on_crosshair=lambda event: self._enqueue_panel_crosshair(
                panel_id, host, event, frozen=frozen
            ),
        )
        source.subscribe_observation(
            lambda observation: self._enqueue_panel_editor_observation(
                panel_id, host, frozen, observation
            )
        )
        source.subscribe_viewport_observation(
            lambda observation: self._enqueue_panel_editor_viewport(
                panel_id, host, frozen, observation
            )
        )
        source.subscribe_focus_observation(
            lambda focused_index, subject, generation, revision: (
                self._enqueue_panel_focus(
                    panel_id,
                    host,
                    focused_index,
                    subject,
                    generation,
                    revision,
                    frozen=frozen,
                )
            )
        )
        return source

    def _release_panel_editor(self, binding: PanelBinding) -> None:
        """Detach and close Edit's subscription and frozen plotting host."""

        host = binding.editor_host
        selections = binding.editor_selections
        pending_entry = binding.editor_configuration
        binding.editor_configuration = None
        binding.editor_host = None
        binding.editor_selections = None
        mount = getattr(self.view, "show_panel_editor", None)
        if callable(mount) and host is not None:
            mount(binding.panel_id, None)
        try:
            if pending_entry is not None:
                candidate, pending, _normalize, _target, _frozen = pending_entry
                pending.cancel()
                if candidate is not host:
                    self._retire_plot_host(candidate)
            if selections is not None:
                selections.close()
        finally:
            if host is not None:
                self._retire_plot_host(host)

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
        """Refresh Edit from the next accepted canonical presentation."""

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return False
        front = self.session.signal_plane.freeze()
        value = front.value(binding.state.signal)
        publication = front.publication(binding.state.signal)
        if value is None or publication is None:
            self._report(
                f"{binding.state.signal} has not published yet",
                severity="warning",
            )
            return False
        surface = binding.accepted_surface
        shown_publication = None if surface is None else surface.publication
        shown_input = None if surface is None else surface.plot_input
        if shown_publication is publication and shown_input is not None:
            previous = binding.frozen_data
            surface = binding.accepted_surface
            if surface is None or surface.description is None:
                return False
            frozen = self._panel_frozen_data(
                binding,
                publication=publication,
                plot_input=shown_input,
                event_records=surface.event_records,
                target=binding.state,
                description=surface.description,
            )
            binding.frozen_data = frozen
            binding.refresh_requested = False
            if binding.editor_host is not None:
                try:
                    if previous is not None and previous.plot_input is shown_input:
                        self._refresh_panel_editor_selection(binding)
                    else:
                        self._replace_panel_editor_host(binding)
                except Exception as error:
                    # Frozen record and Frozen pixels are one transaction.
                    # Restore the previous record if its replacement host
                    # could not be mounted; never save new bytes through an
                    # old surface merely because both belong to one run.
                    binding.frozen_data = previous
                    self._report(
                        f"cannot mount {binding.state.title} plot "
                        f"editor: {_error_text(error)}",
                        severity="error",
                    )
            self.refresh_panel_editor(panel_id)
            return True
        # A newer exact publication exists.  The board tick only submits its
        # canonical projection; materialization and rendering remain on the
        # board/plot workers, and _panel_presented installs the frozen record
        # when that same-shot surface is accepted.
        binding.refresh_requested = True
        self.board.tick()
        return True

    def save_panel_figure(self, panel_id: str, selected: str) -> bool:
        """Submit one exact frozen Panel Save without blocking the owner."""

        key = str(panel_id)
        binding = self.panels.get(key)
        if binding is None:
            return False
        if self._closing:
            self._report("Console is closing; Panel Save was not started", severity="warning")
            return False
        if key in self._saving_panels:
            self._report(f"{binding.state.title} is already saving", severity="warning")
            return False
        frozen = binding.frozen_data
        if frozen is None:
            self._report(
                f"{panel_id} has no frozen data to save",
                severity="warning",
            )
            return False
        if binding.frozen_stale or frozen.signal != binding.state.signal:
            self._report(
                f"{panel_id} frozen surface is stale; Refresh it before Save",
                severity="warning",
            )
            return False
        selected = str(selected).strip()
        if not selected:
            return False

        # Everything below is frozen on the owner turn.  The worker never
        # reads the live binding again, so Refresh/Edit cannot change a save
        # that is already archive-first in flight.
        state = binding.state
        def work() -> object:
            return _save_panel_figure(
                selected,
                state=state,
                frozen=frozen,
            )

        title = state.title

        def finished(written: object) -> None:
            self._saving_panels.discard(key)
            self._report(
                f"panel saved to {written.image.name} and {written.archive.name}",
                severity="task",
            )
            if self._closing and self._request_close is not None:
                self._request_close()

        def failed(error: BaseException) -> None:
            self._saving_panels.discard(key)
            self._report(
                f"cannot save {title}: {_error_text(error)}",
                severity="error",
            )
            if self._closing and self._request_close is not None:
                self._request_close()

        self._saving_panels.add(key)
        self._report(f"saving {title}", severity="task")
        try:
            self._run_off_thread(work, finished, failed)
        except BaseException as error:
            failed(error)
            return False
        return True

    def restart_panel_producer(self, panel_id: str) -> bool:
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

    def _paints_image_surfaces(
        self,
        binding: PanelBinding,
        state: PanelState | None = None,
    ) -> bool:
        """Whether this panel's surfaces ARE images, as the data decided.

        The authored cell kind cannot answer it: an empty one means the data
        decides, and answering "probably image" there offered an Overlay row
        on a panel that then painted curves.  A grid is a layout -- its CELLS
        are the pictures -- so the question is about the resolved cell, which
        only the bound snapshot can settle.  Views read this; they do not
        re-derive it.
        """

        spec = self._panel_accepted_spec(binding) or self._panel_resolved_spec(
            binding, state
        )
        return False if spec is None else paints_image_surface(spec)

    def _shown_snapshot(self, binding: PanelBinding) -> object | None:
        """The dataset this panel is drawing, live or frozen."""

        surface = binding.accepted_surface
        shown = None if surface is None else surface.plot_input
        snapshot = getattr(shown, "snapshot", shown)
        if getattr(snapshot, "block", None) is not None:
            return snapshot
        if binding.frozen_data is not None:
            return binding.frozen_data.snapshot
        return None

    def _publish_panel_state(self, binding: PanelBinding) -> None:
        """Purely project the current panel records to every view."""

        surface = dict(binding.parameter_surface)
        science_locked = self._task_science_locked(binding)
        surface["science_locked"] = science_locked
        surface["paints_images"] = self._paints_image_surfaces(binding)
        for section in ("semantic", "display", "fit"):
            authored = dict(getattr(binding.state, section))
            declared = tuple(surface.get(section, ()))
            legal: dict[str, object] = {}
            for field in declared:
                key = str(field["key"])
                if key not in authored:
                    continue
                value = authored[key]
                choices = tuple(field.get("choices") or ())
                if not choices or (
                    value is None and bool(field.get("allow_none"))
                ) or any(value == choice for _label, choice in choices):
                    legal[key] = value
            surface[section] = tuple(
                {
                    **dict(field),
                    "value": legal.get(str(field["key"]), field.get("value")),
                }
                for field in declared
            )
        snapshot = self._shown_snapshot(binding)
        schema = getattr(getattr(snapshot, "block", None), "schema", None)
        if schema is None:
            schema = self._panel_schema(binding)
        surface.update(
            {"data_structure": (), "data_scope": ()}
            if schema is None
            else panel_data_shape(schema, binding.accepted_display)
        )
        binding.parameter_surface = surface

        set_projection = getattr(self.view, "set_panel_projection", None)
        if callable(set_projection):
            set_projection(
                binding.panel_id,
                binding.state,
                binding.parameter_surface,
            )
        self.view.set_panel_selectors_enabled(
            binding.panel_id,
            self._deriving,
        )
        self._offer_panel(binding.panel_id)
        self.refresh_panel_editor(binding.panel_id)
        self.refresh_panel_publisher_editor(binding.panel_id)

    def _release_panel(self, binding: PanelBinding) -> None:
        """Let go of one panel's derivation and its plotting host."""

        self._release_panel_history(binding)
        if binding.selections is not None:
            binding.selections.close()
        if binding.bridge is not None:
            binding.bridge.close()
        binding.bridge = binding.selections = None
        self._cancel_panel_configuration(binding)
        binding.refresh_requested = False
        host = binding.host
        port = binding.port
        binding.port = None
        if port is not None:
            port.close()
        if host is not None:
            self._retire_plot_host(host)

    def _retire_plot_host(self, host: object) -> None:
        """Request worker shutdown without waiting in a GUI mutation."""

        if host.close(timeout=0.0):
            return
        if not any(current is host for current in self._retired_plot_hosts):
            self._retired_plot_hosts.append(host)

    def _poll_retired_plot_hosts(self) -> None:
        """Forget superseded hosts only after their existing worker has stopped."""

        self._retired_plot_hosts[:] = [
            host
            for host in self._retired_plot_hosts
            if not host.close(timeout=0.0)
        ]

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

    def _remove_panel_now(self, panel_id: str) -> bool:
        """Retire one panel after its caller has passed command admission."""

        key = str(panel_id)
        binding = self.panels.pop(key, None)
        if binding is None:
            return False
        self._release_panel_editor(binding)
        close_editor = getattr(self.view, "close_panel_editor", None)
        if callable(close_editor):
            close_editor(key)
        close_publisher = getattr(self.view, "close_panel_publisher_editor", None)
        if callable(close_publisher):
            close_publisher(key)
        self._release_panel(binding)
        self.view.remove_panel(key)
        for previews in self._auto_task_previews.values():
            previews.pop(key, None)
        return True

    def remove_panel(self, panel_id: str) -> None:
        if self._remove_panel_now(panel_id):
            self._refresh_console_projection()

    # ------------------------------------------------------------------ running

    def set_paused(self, paused: bool) -> None:
        """The presenter owns the answer; the window is told what it now is."""

        self._paused = bool(paused)
        self.view.set_paused(self._paused)
        self._refresh_console_projection()

    def set_deriving(self, deriving: bool) -> None:
        """Whether plot surfaces own pointer input.

        Off rejects area, zoom, pan, hover and facet focus; the ordinary wheel
        remains with the board.  On gives the focused/non-grid surface its
        selector gestures while a FacetGrid overview remains focus-only.
        """

        self._deriving = bool(deriving)
        self.view.set_selectors(self._deriving)
        for panel_id, binding in self.panels.items():
            self._apply_deriving(binding)
            # The card applies the same global pointer gate to its plot.
            self.view.set_panel_selectors_enabled(
                panel_id,
                self._deriving,
            )
        self._report(
            "selectors enabled" if self._deriving else "selectors disabled",
            severity="task",
        )

    def beat(self) -> None:
        """Advance lifecycle always; Pause freezes only the Monitor tick.

        The commit runs even while paused: Pause stops NEW shots from being
        staged, but a batch already travelling must still land -- atomically,
        as one group -- or pausing at the wrong moment would freeze half a
        causal group one shot behind the other half.
        """

        self._drain_panel_interactions()
        self._poll_retired_plot_hosts()
        if self._closing:
            self.board.commit(admit_new=False)
            self.poll_logic()
            self._advance_close()
            return
        self._settle_panel_hosts()
        if not self._paused:
            self.board.tick()
        self.board.commit(admit_new=not self._paused)
        self._report_panel_errors()
        self.poll_logic()
        self._refresh_signal_choices()

    def commit_surfaces(self) -> None:
        """The completion-driven owner turn: commit what finished, NOW.

        Driven by the board's wake through the composition root's GUI-thread
        relay, so a group's present latency is its slowest member plus one
        queued hop -- not the remainder of the current display beat.  Runs
        during Pause for the same reason the beat's commit does.
        """

        # Plot callbacks carry semantic owner work as well as completed
        # surfaces.  Settle that work first: an Area release must become the
        # canonical PanelState before a Fit click can read/configure it, and a
        # viewport or threshold must not wait for the periodic display beat.
        self._drain_panel_interactions()
        self._settle_panel_hosts()
        self.board.commit(admit_new=not self._paused and not self._closing)
        self._report_panel_errors()
        if self._closing:
            self._poll_retired_plot_hosts()
            self.poll_logic()
            self._advance_close()

    def _enqueue_panel_interaction(self, interaction: Callable[[], None]) -> None:
        self._panel_interactions.put(interaction)
        self.board.wake.request_owner_wake()

    def _drain_panel_interactions(self) -> None:
        while True:
            try:
                interaction = self._panel_interactions.get_nowait()
            except Empty:
                return
            try:
                interaction()
            except (TypeError, ValueError) as error:
                self._report(
                    f"cannot apply panel interaction: {_error_text(error)}",
                    severity="error",
                )

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
                    binding.auto_preview,
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
                    interval_ms=self._panel_interval(saved.interval_ms),
                )
                used_titles.add(state.title)
                binding = PanelBinding(
                    panel_id,
                    state,
                    parameter_surface=self._unbound_panel_parameters(state),
                )
                panels.append(binding)
                if not state.signal:
                    continue
                value = front.value(state.signal)
                if value is None:
                    missing.append(state.signal)
                    continue
                fitting = self._fitting_cell_kind(
                    value,
                    state.kind,
                    state.cell_kind,
                )
                if fitting is None:
                    incompatible.append((state.signal, state.kind))
                    continue
                binding.state = state
                binding.parameter_surface = self._unbound_panel_parameters(state)
                binding.port = self._make_panel_port(binding)
        except Exception as error:
            for binding in panels:
                self._release_panel(binding)
            raise LayoutError(
                f"cannot prepare the layout panels: {_error_text(error)}"
            ) from error
        return _LayoutCandidate(
            resolved.logic,
            tuple(panels),
            serial,
            tuple(missing),
            tuple(incompatible),
        )

    def _build_layout_candidate(self, document: LayoutDocument) -> _LayoutCandidate:
        try:
            for state in document.panels:
                task_console_panel_identity(state.kind, state.cell_kind)
        except (TypeError, ValueError) as error:
            raise LayoutError(_error_text(error)) from error
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
            self.view.add_logic_row(
                binding.node_id, kind, binding.descriptor.offers_a_preview
            )
        for binding in candidate.panels:
            self.panels[binding.panel_id] = binding
            self.view.add_panel(binding.panel_id, binding.state.title)
            self.view.show_panel(binding.panel_id, None)
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
            self._report(
                f"cannot load the layout: {_error_text(error)}", severity="error"
            )
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
            "Save TaskConsole layout",
            # A name, not an empty box: a board is a thing you keep and
            # reload, so it is named plainly and the dialog warns before it
            # replaces one.
            str(Path(self.session.day_folder()) / "layout.json"),
            "Layouts (*.json)",
        )
        if not path:
            return ""
        try:
            self._layout_document().write(path)
        except Exception as error:
            self._report(
                f"cannot save the layout: {_error_text(error)}", severity="error"
            )
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
            self._report(
                f"cannot read that layout: {_error_text(error)}", severity="error"
            )
            return False
        return self.apply_layout(document)

    def save_screenshot(self) -> str:
        """Save one ordinary image of the whole current TaskConsole GUI."""

        path = self.view.ask_save_path(
            "Save TaskConsole screenshot",
            str(Path(self.session.day_folder()) / "console.png"),
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
            self._report(
                f"cannot save screenshot: {_error_text(error)}", severity="error"
            )
            return ""
        self._report(f"screenshot saved to {written}", severity="task")
        return str(written)

    def _apply_deriving(self, binding: PanelBinding) -> None:
        """Attach the panel's one ROI/fit publication bridge when ready."""

        if (
            binding.host is None
            or binding.bridge is not None
            or not hasattr(binding.host, "subscribe_selection")
        ):
            return
        if binding.accepted_display is None:
            return
        initial_selection = panel_selection_from_document(binding.state.selector)
        initial_publication = None
        bridge_selection = (
            initial_selection
            if initial_selection is not None
            and panel_selection_derives_signal(initial_selection)
            else None
        )
        if bridge_selection is not None:
            initial_publication = binding.display_publication
            if initial_publication is None:
                initial_publication = binding.display_publication
            if initial_publication is None:
                # The host may render before its first board acceptance.  Wait
                # for that exact publication rather than restoring from latest.
                return
        source_host = binding.host
        binding.bridge, binding.selections = attach_selection_bridge(
            self.session.signal_plane,
            source_host,
            binding.signal,
            bridge_id=binding.panel_id,
            # The panel's port holds a fit's exact parent publication
            # (pending or presented) at accept time; resolve lazily so a
            # port attached after this bridge still answers.
            source_publication_for=lambda generation, revision: (
                None
                if binding.port is None
                else binding.port.publication_for_identity(generation, revision)
            ),
            request_owner_wake=self.board.wake.request_owner_wake,
            initial_selection=bridge_selection,
            initial_publication=initial_publication,
            on_observation=lambda bridge, observation, publication: (
                self._enqueue_panel_observation(
                    binding.panel_id,
                    source_host,
                    bridge,
                    observation,
                    publication,
                )
            ),
            on_threshold=lambda event, host=binding.host: (
                self._enqueue_panel_threshold(binding.panel_id, host, event)
            ),
            on_crosshair=lambda event, host=binding.host: (
                self._enqueue_panel_crosshair(binding.panel_id, host, event)
            ),
        )
        binding.selections.subscribe_viewport_observation(
            lambda observation: self._enqueue_panel_viewport(
                binding.panel_id, source_host, observation
            )
        )
        binding.selections.subscribe_focus_observation(
            lambda focused_index, subject, generation, revision: (
                self._enqueue_panel_focus(
                    binding.panel_id,
                    source_host,
                    focused_index,
                    subject,
                    generation,
                    revision,
                )
            )
        )
        binding.bridge.configure_outputs(binding.state.published_outputs)

    def _enqueue_panel_threshold(
        self,
        panel_id: str,
        host: object,
        observation: object,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._settle_panel_threshold(
                panel_id, host, observation, frozen=frozen
            )
        )

    def _enqueue_panel_crosshair(
        self,
        panel_id: str,
        host: object,
        event: object,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._settle_panel_crosshair(
                panel_id, host, event, frozen=frozen
            )
        )

    def _settle_panel_crosshair(
        self,
        panel_id: str,
        source: object,
        event: object,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> None:
        """A crosshair placed on either surface is the panel's marker.

        Both of a panel's views look at the same experiment, so they point
        at the same place: the marker lands in the panel record (a board
        restores it) and mirrors to the sibling surface.  Removing it
        clears both.  A stale Edit surface neither speaks nor listens,
        exactly as every other gesture.
        """

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return
        accepted = self._accepted_panel_interaction(
            binding,
            source,
            event,
            frozen=frozen,
            exact_subject=False,
        )
        if accepted is None:
            return
        change = SelectionChange(
            str(getattr(event.change, "value", event.change))
        )
        if change is SelectionChange.REMOVED:
            document: dict[str, object] = {}
        else:
            value = event.selector.value
            document = {"x": float(value.x), "y": float(value.y)}
        if dict(binding.state.crosshair) == document:
            return
        self._remember_panel_view(binding, crosshair=document)
        for host in (binding.host, binding.editor_host):
            if (
                host is None
                or host is source
                or (host is binding.editor_host and binding.frozen_stale)
            ):
                continue
            if document:
                host.set_crosshair_selector(document["x"], document["y"])
            else:
                try:
                    host.selector_state(SelectorKind.CROSSHAIR)
                except KeyError:
                    continue
                host.remove_selector(SelectorKind.CROSSHAIR)
        self._track_panel_configuration(
            binding,
            source,
            source.describe_display(),
        )

    def _accepted_panel_interaction(
        self,
        binding: PanelBinding,
        source: object,
        observation: object | None,
        *,
        frozen: PanelFrozenData | None = None,
        exact_subject: bool = True,
        subject: object | None = None,
        data_generation: object = None,
        data_revision: object = None,
    ) -> tuple[object, object] | None:
        """Validate one gesture against its exact current surface."""

        publication = None
        plot_input = None
        if source is binding.host:
            surface = binding.accepted_surface
            if surface is not None and surface.host is source:
                publication = surface.publication
                plot_input = surface.plot_input
        elif (
            source is binding.editor_host
            and frozen is not None
            and binding.frozen_data is not None
            and binding.frozen_data.publication is frozen.publication
            and binding.frozen_data.snapshot.ref == frozen.snapshot.ref
            and not binding.frozen_stale
        ):
            publication = frozen.publication
            plot_input = frozen.plot_input
        if observation is None:
            identity_matches = plot_identity_matches_plot_input(
                plot_input,
                data_generation,
                data_revision,
            )
        else:
            identity_matches = observation_matches_plot_input(
                observation,
                plot_input,
            )
            subject = getattr(observation, "subject", None)
        if (
            publication is None
            or plot_input is None
            or not identity_matches
        ):
            return None
        description = self._panel_accepted_display(binding, source)
        if (
            description is None
            or subject is None
            or semantic_spec(description.spec).kind is not subject.plot_kind
            or exact_subject
            and description.selection_subject != subject
        ):
            return None
        return publication, description

    def _settle_panel_threshold(
        self,
        panel_id: str,
        source: object,
        observation: object,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> None:
        """A threshold set on either surface is the panel's answer.

        Both of a panel's views classify the same experiment, so they cannot
        classify it at different levels: the fitted populations and the
        fidelity printed beside them would disagree with no way to tell which
        is current.  Removing it hands both views back to their own fit --
        which may legitimately differ, because it is measured from the data
        each of them holds.
        """

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return
        accepted = self._accepted_panel_interaction(
            binding,
            source,
            observation,
            frozen=frozen,
            exact_subject=False,
        )
        if accepted is None:
            return
        _publication, description = accepted
        if not accepts_classifier_thresholds(
            description.spec,
            description.display_state.values,
        ):
            return
        target = tuple(observation.classifier_thresholds)
        if binding.state.classifier_thresholds == target:
            return
        self._remember_panel_view(
            binding, classifier_thresholds=target
        )
        for host in (binding.host, binding.editor_host):
            if (
                host is None
                or host is source
                or (host is binding.editor_host and binding.frozen_stale)
            ):
                continue
            other = self._panel_accepted_display(binding, host)
            if other is None or not accepts_classifier_thresholds(
                other.spec,
                other.display_state.values,
            ):
                continue
            operation = host.configure(classifier_thresholds=target)
            self._track_panel_configuration(binding, host, operation)
        self._track_panel_configuration(
            binding,
            source,
            source.describe_display(),
        )

    def _enqueue_panel_observation(
        self,
        panel_id: str,
        source: object,
        bridge: object,
        observation: object,
        publication: object,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._route_panel_observation(
                panel_id, source, bridge, observation, publication
            )
        )

    def _enqueue_panel_editor_observation(
        self,
        panel_id: str,
        host: object,
        frozen: PanelFrozenData,
        observation: object,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._route_panel_editor_observation(
                panel_id, host, frozen, observation
            )
        )

    def _enqueue_panel_viewport(
        self,
        panel_id: str,
        source: object,
        observation: object,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._route_panel_viewport(
                panel_id, source, observation
            )
        )

    def _enqueue_panel_editor_viewport(
        self,
        panel_id: str,
        host: object,
        frozen: PanelFrozenData,
        observation: object,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._route_panel_editor_viewport(
                panel_id, host, frozen, observation
            )
        )

    def _enqueue_panel_focus(
        self,
        panel_id: str,
        host: object,
        focused_index: int | None,
        subject: object,
        data_generation: object,
        data_revision: object,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> None:
        self._enqueue_panel_interaction(
            lambda: self._route_panel_focus(
                panel_id,
                host,
                focused_index,
                subject,
                data_generation,
                data_revision,
                frozen=frozen,
            )
        )

    def _synchronize_panel_interaction(
        self,
        binding: PanelBinding,
        other_host: object | None,
        selection: object | None,
        viewport: object,
    ) -> object:
        """Keep live and frozen views on one selector/viewport truth."""

        def mirror(operation: object) -> None:
            # The live panel's widget stages its fronts, so a selector mirrored
            # onto it must be presented when drawn; the frozen Edit surface
            # presents its own fronts.
            if other_host is binding.host:
                self._present_when_done(binding, operation)

        if viewport is not _UNCHANGED:
            # Zoom and pan are how an operator looks, not what they ask for.
            # A viewport used to be routed to the producer whenever no region
            # was drawn, so moving the view re-pointed the hardware; and a
            # region taken away then routed the viewport in its place instead
            # of putting back what the region had overwritten.
            if (
                binding.interaction_viewport is not None
                and binding.interaction_viewport[1] == viewport
            ):
                return _UNCHANGED
            binding.interaction_viewport = (self._panel_view_identity(binding), viewport)
            if other_host is not None:
                self._track_panel_configuration(
                    binding,
                    other_host,
                    other_host.configure(viewport=viewport),
                )
            return _UNCHANGED

        if selection is None:
            previous = panel_selection_from_document(binding.state.selector)
            if previous is None:
                return _UNCHANGED
            self._remember_panel_view(binding, selector={})
            if other_host is not None:
                mirror(_remove_panel_selection(other_host, previous))
            if not self._task_science_locked(binding):
                self._resync_producer_draft(binding, previous)
            return _UNCHANGED

        remembered = panel_selection_from_document(binding.state.selector)
        if remembered is not None and _same_panel_selection(remembered, selection):
            return _UNCHANGED
        self._remember_panel_view(
            binding, selector=panel_selection_document(selection)
        )
        if other_host is not None:
            mirror(_apply_panel_selection(other_host, selection))
        return selection

    def _route_panel_viewport(
        self,
        panel_id: str,
        source: object,
        observation: object,
    ) -> None:
        binding = self.panels.get(str(panel_id))
        if binding is None or self._accepted_panel_interaction(
            binding, source, observation
        ) is None:
            return
        self._synchronize_panel_interaction(
            binding,
            None if binding.frozen_stale else binding.editor_host,
            _UNCHANGED,
            observation.display,
        )
        self._track_panel_configuration(
            binding,
            source,
            source.describe_display(),
        )

    def _route_panel_focus(
        self,
        panel_id: str,
        source: object,
        focused_index: int | None,
        subject: object,
        data_generation: object,
        data_revision: object,
        *,
        frozen: PanelFrozenData | None = None,
    ) -> None:
        binding = self.panels.get(str(panel_id))
        if binding is None:
            return
        accepted = self._accepted_panel_interaction(
            binding,
            source,
            None,
            frozen=frozen,
            exact_subject=False,
            subject=subject,
            data_generation=data_generation,
            data_revision=data_revision,
        )
        if accepted is None or accepted[1].kind is not PlotKind.FACET_GRID:
            return
        focus = focused_index
        if focus == binding.state.focused_cell:
            return
        previous = panel_selection_from_document(binding.state.selector)
        changes: dict[str, object] = {"focused_cell": focus}
        if previous is not None and not panel_selection_matches_subject(
            previous,
            subject,
        ):
            changes["selector"] = {}
        binding.interaction_viewport = None
        self._remember_panel_view(binding, **changes)
        if "selector" in changes and binding.bridge is not None:
            binding.bridge.clear_selection()
        other = (
            binding.editor_host if source is binding.host else binding.host
        )
        if other is not None and not (
            other is binding.editor_host and binding.frozen_stale
        ):
            operation = other.configure(facet_focus=focus)
            self._track_panel_configuration(binding, other, operation)
        # Focus changes the accepted selection subject without changing the
        # PlotSpec.  Refresh the existing description through the same
        # configure-accept slots; no polling or parallel focus state remains.
        self._track_panel_configuration(
            binding,
            source,
            source.describe_display(),
        )

    def _apply_selection_observation(
        self,
        binding: PanelBinding,
        observation: object,
        publication: object,
        *,
        other_host: object | None,
        expected_snapshot: object | None = None,
    ) -> None:
        removed = observation.change is SelectionChange.REMOVED
        selection = None
        if not removed:
            remembered = panel_selection_from_document(binding.state.selector)
            if remembered is None or not _same_panel_selection(
                remembered, observation.state
            ):
                binding.selection_revision = max(
                    binding.selection_revision + 1,
                    int(observation.state.revision),
                )
            selection = replace(
                observation.state,
                revision=binding.selection_revision,
            )
        synchronized = self._synchronize_panel_interaction(
            binding,
            other_host,
            selection,
            _UNCHANGED,
        )
        bridge = binding.bridge
        if removed:
            if bridge is not None:
                bridge.clear_selection()
        elif synchronized is not _UNCHANGED:
            assert selection is not None
            if bridge is not None:
                if panel_selection_derives_signal(selection):
                    bridge.commit_selection(
                        selection,
                        source_publication=publication,
                    )
                else:
                    bridge.clear_selection()
        if synchronized is _UNCHANGED or self._task_science_locked(binding):
            return
        self._route_exact_panel_selection(
            binding.panel_id,
            binding.state.signal,
            publication,
            synchronized,
            expected_snapshot=expected_snapshot,
        )

    def _route_panel_observation(
        self,
        panel_id: str,
        source: object,
        bridge: object,
        observation: object,
        publication: object,
    ) -> None:
        binding = self.panels.get(str(panel_id))
        if (
            binding is None
            or binding.host is not source
            or binding.bridge is not bridge
        ):
            return
        accepted = self._accepted_panel_interaction(
            binding, source, observation
        )
        if accepted is None or accepted[0] is not publication:
            return
        if not panel_selection_matches_subject(
            observation.state, observation.subject
        ):
            return
        self._apply_selection_observation(
            binding,
            observation,
            publication,
            other_host=binding.editor_host,
        )
        self._track_panel_configuration(
            binding,
            source,
            source.describe_display(),
        )

    def _route_panel_editor_viewport(
        self,
        panel_id: str,
        host: object,
        frozen: PanelFrozenData,
        observation: object,
    ) -> None:
        binding = self.panels.get(str(panel_id))
        if binding is None or self._accepted_panel_interaction(
            binding,
            host,
            observation,
            frozen=frozen,
        ) is None:
            return
        self._synchronize_panel_interaction(
            binding,
            binding.host,
            _UNCHANGED,
            observation.display,
        )
        self._track_panel_configuration(
            binding,
            host,
            host.describe_display(),
        )

    def _route_panel_editor_observation(
        self,
        panel_id: str,
        host: object,
        frozen: PanelFrozenData,
        observation: object,
    ) -> None:
        """Accept a region only from this exact still-current Frozen input."""

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return
        accepted = self._accepted_panel_interaction(
            binding,
            host,
            observation,
            frozen=frozen,
        )
        if accepted is None or not panel_selection_matches_subject(
            observation.state, observation.subject
        ):
            return
        publication, _description = accepted
        self._apply_selection_observation(
            binding,
            observation,
            publication,
            other_host=binding.host,
            expected_snapshot=frozen.snapshot,
        )
        self._track_panel_configuration(
            binding,
            host,
            host.describe_display(),
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
        snapshot = expected_snapshot
        if snapshot is None:
            binding = self.panels.get(str(panel_id))
            surface = None if binding is None else binding.accepted_surface
            if (
                surface is None
                or surface.publication is not publication
            ):
                raise RuntimeError(
                    f"{panel_id} selection is not on its accepted publication"
                )
            shown = surface.plot_input
            snapshot = getattr(shown, "snapshot", shown)
        if (
            expected_snapshot is not None
            and getattr(snapshot, "ref", None)
            != getattr(expected_snapshot, "ref", None)
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

        context = self._selection_context(publication)
        if str(getattr(selection, "selector_kind", "")) != "area":
            # A region is an area of the data; that is the only gesture whose
            # coordinates mean a producer's setting.  A threshold line, an
            # x range or a crosshair say something about the reading, not
            # about how to take the next one.
            return
        draft = dict(producer.draft.values)
        patch = producer.descriptor.selection_patch(
            selection,
            draft=draft,
            context=context,
        )
        if patch is not None:
            self.update_logic_draft(producer_node_id, values=patch)

    def _resync_producer_draft(
        self, binding: PanelBinding, selection: object
    ) -> None:
        """Show what the producer is actually set to, once its region is gone.

        Not what its fields held before the region overwrote them: those were
        replaced by a deliberate gesture, and a run has happened since with
        the values the gesture asked for.  Restoring the earlier ones would
        mean that cancelling a region silently re-points the hardware on the
        next Start -- an operator who removes an ROI and presses Start expects
        nothing to change.  So the fields the region owns are filled from the
        run's own readback, and the descriptor -- which alone knows which
        fields those are and how a device reports them -- provides both.
        """

        publication = binding.display_publication
        if publication is None:
            return
        producer_node_id = self._direct_producer_node_id(binding.state.signal)
        if producer_node_id is None:
            return
        producer = self.logic.get(producer_node_id)
        if producer is None:
            return
        applied = producer.descriptor.applied_selection_values(
            selection, context=self._selection_context(publication)
        )
        if applied:
            self.update_logic_draft(producer_node_id, values=dict(applied))

    def _selection_context(self, publication: object) -> dict[str, Any]:
        """Public run-time device readback, as data only."""

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
        return context

    def _panel_view_identity(
        self,
        binding: PanelBinding,
        *,
        state: PanelState | None = None,
        subject: object | None = None,
    ) -> object:
        """What a remembered viewport was measured on.

        Dataset identity alone is insufficient: two plots over the same bytes
        can put different axes under the same numeric rectangle.  The resolved
        spec and coordinate units are therefore part of the identity too.
        """

        selected = binding.state if state is None else state
        snapshot = self._shown_snapshot(binding) if subject is None else subject
        snapshot = getattr(snapshot, "snapshot", snapshot)
        block = getattr(snapshot, "block", None)
        schema = getattr(block, "schema", None)
        description = (
            binding.accepted_display
            if state is None and subject is None
            else None
        )
        if description is None:
            resolved = self._panel_resolved_spec(
                binding, selected, subject=snapshot
            )
            display = selected.display
            focused_cell = selected.focused_cell
        else:
            resolved = description.spec
            display = description.display_state.values
            focused_cell = description.facet_focus
        units = tuple(
            (name, display.get(name))
            for name in (
                "x_display_unit",
                "y_display_unit",
                "facet_display_unit",
            )
            if name in display
        )
        return (
            (
                None
                if snapshot is None
                else getattr(
                    getattr(
                        getattr(snapshot, "ref", None),
                        "stream_generation",
                        None,
                    ),
                    "value",
                    None,
                )
            ),
            None if schema is None else getattr(schema, "fingerprint", None),
            resolved,
            units,
            focused_cell,
        )

    def _report_panel_errors(self) -> None:
        """Say what a gesture could not do, once, where an operator looks.

        These arrive from inside a plot callback whose exceptions are swallowed
        by design.  Unreported, the panel simply stops answering boxes.
        """

        for panel_id, binding in self.panels.items():
            error = (
                getattr(binding.editor_selections, "last_error", None)
                or getattr(binding.selections, "last_error", None)
                # The derivation bridge records its own refusals (a failed
                # ROI/fit publication); nothing read them, so a bridge-side
                # failure left the derived signal silently absent.
                or getattr(binding.bridge, "last_error", None)
                or getattr(binding.port, "last_error", None)
            )
            if error is None:
                # The mark is the panel's CURRENT condition, not a log of what
                # once went wrong: a panel that has drawn again since clears
                # its own error, and the card that still wore the dot told an
                # operator a healthy panel was broken until the window closed.
                if binding.reported_error is not None:
                    binding.reported_error = None
                    if panel_id in self.view.panel_ids():
                        self.view.set_panel_status(panel_id, "", error=False)
                continue
            if error is binding.reported_error:
                continue
            binding.reported_error = error
            self._report(
                f"{binding.title}: {_error_text(error)}", severity="error"
            )
            # A refused projection is a STATE, not a reason to have no form:
            # the semantic fates that fix it stay editable.
            self._degrade_panel_surface(binding, error)
            # And on the card itself, which has a status line nothing wrote to.
            # A board-wide line says which panel; the panel says it is the one.
            if panel_id in self.view.panel_ids():
                self.view.set_panel_status(panel_id, _error_text(error), error=True)

    def _panel_plot_error(self, panel_id: str, message: str) -> None:
        """Report one refusal a mounted plot widget raised through its card."""

        binding = self.panels.get(str(panel_id))
        title = binding.title if binding is not None else str(panel_id)
        self._report(f"{title}: {message}", severity="warning")

    def _report(self, text: str, *, severity: str) -> None:
        show = getattr(self.view, "show_status", None)
        if show is not None:
            show(text, severity)


    # ------------------------------------------------------------------- logic

    @staticmethod
    def _is_task(binding: LogicBinding) -> bool:
        kind = getattr(binding.descriptor, "kind", "")
        return str(getattr(kind, "value", kind)) == "task"

    def _active_task(self) -> LogicBinding | None:
        task_id = self._active_task_id
        return None if task_id is None else self.logic.get(task_id)

    def _task_protected_signals(self) -> frozenset[str]:
        active = self._active_task()
        if active is None:
            return frozenset()
        names = set(
            ()
            if active.host is None
            else active.host.published_signals()
        )
        for preview in active.preview_specs:
            producer = (
                active.node_id
                if not preview.producer
                else f"{active.node_id}/{preview.producer}"
            )
            names.add(stable_signal_key(producer, preview.output.name))
            if preview.overlay is not None:
                names.add(stable_signal_key(producer, preview.overlay.name))
        return frozenset(names)

    def _task_science_locked(self, binding: PanelBinding) -> bool:
        protected = self._task_protected_signals()
        return bool(
            protected
            and (
                binding.state.signal in protected
                or binding.state.overlay_signal in protected
            )
        )

    def _task_panel_science_blocked(
        self,
        binding: PanelBinding,
        action: str,
    ) -> bool:
        if not self._task_science_locked(binding):
            return False
        active = self._active_task()
        assert active is not None
        self._report(
            f"{active.node_id}: preview science identity is frozen while "
            f"the Task runs; use Stop task before {action}",
            severity="task",
        )
        return True

    def _task_command_blocked(self, action: str, *, node_id: str = "") -> bool:
        """Reject Logic/hardware identity changes while one Task is active."""

        active = self._active_task()
        if active is None:
            return False
        del node_id
        _state, status = self._logic_state(active)
        self._report(
            f"{active.node_id}: {status}; use Stop task before {action}",
            severity="task",
        )
        return True

    def _project_task_takeover(self) -> None:
        active = self._active_task()
        takeover = active is not None
        if takeover != self._shown_task_takeover:
            self.view.set_task_takeover(takeover)
            self._shown_task_takeover = takeover
            for panel in tuple(self.panels.values()):
                self._publish_panel_state(panel)
            if active is not None:
                _state, status = self._logic_state(active)
                self._report(f"{active.node_id}: {status}", severity="task")

    def _begin_task_takeover(self, binding: LogicBinding) -> None:
        if not self._is_task(binding):
            return
        active = self._active_task()
        if active is not None and active is not binding:
            raise RuntimeError("TaskConsole already has an active Task")
        self._active_task_id = binding.node_id
        self._project_task_takeover()

    def stop_active_task(self) -> bool:
        """Route the status-strip action through the ordinary Stop endpoint."""

        active = self._active_task()
        return False if active is None else self.stop_logic(active.node_id)

    def set_logic_auto_preview(self, node_id: str, enabled: bool) -> None:
        """Remember whether starting this node should open its plot panel."""

        binding = self.logic.get(str(node_id))
        if binding is None:
            return
        if self._task_command_blocked("changing preview policy"):
            return
        binding.auto_preview = bool(enabled)
        self._refresh_console_projection()

    def _ensure_node_previews(self, binding: LogicBinding) -> None:
        """Mount this node's declared previews through the ordinary panel path.

        WHICH panel, and drawn HOW, is the node's own declaration -- it knows
        what it just measured, and a console reading only a shape would be
        guessing; WHETHER it opens is the operator's per-row preference.

        The declaration is the RUNNING node's, frozen when it started: what
        the operator types next belongs to the next run.
        """

        if not binding.auto_preview or binding.host is None:
            return
        previews = tuple(binding.preview_specs)
        if not previews:
            return
        if binding.preview_host is not binding.host:
            # A new run asks the question again; ONE run answers it once, so a
            # preview the operator closed stays closed until the next Start.
            binding.preview_host = binding.host
            binding.previewed = ()
            self._preview_errors[binding.node_id] = set()
        pending = tuple(
            preview
            for preview in previews
            if stable_signal_key(
                (
                    binding.node_id
                    if not preview.producer
                    else f"{binding.node_id}/{preview.producer}"
                ),
                preview.output.name,
            )
            not in binding.previewed
        )
        if not pending:
            return
        front = self.session.signal_plane.freeze()
        for preview in pending:
            output_name = preview.output.name
            producer = (
                binding.node_id
                if not preview.producer
                else f"{binding.node_id}/{preview.producer}"
            )
            signal = stable_signal_key(producer, output_name)
            overlay_signal = (
                ""
                if preview.overlay is None
                else stable_signal_key(producer, preview.overlay.name)
            )
            value = front.value(signal)
            # Nothing published yet is not an answer -- a run that ends in one
            # tick publishes on the same poll that stops it, and gating on
            # "still running" is how that node's plot never appeared.
            if value is None or (
                overlay_signal and front.value(overlay_signal) is None
            ):
                continue
            if any(panel.state.signal == signal for panel in self.panels.values()):
                binding.previewed += (signal,)
                continue
            publication = front.publication(signal)
            # The node names the kind it means; this exact dataset may not be
            # drawable that way is a broken node declaration, not permission
            # for Workbench to silently choose different science semantics.
            kind = str(preview.plot_kind)
            if self._spec_for_value(value, kind, "") is None:
                error_key = f"{signal}|{kind}"
                errors = self._preview_errors.setdefault(binding.node_id, set())
                if error_key not in errors:
                    errors.add(error_key)
                    self._report(
                        f"{binding.node_id}: preview {signal!r} is incompatible "
                        f"with declared plot kind {kind!r}",
                        severity="error",
                    )
                continue
            try:
                panel = self.add_panel(
                    signal,
                    value.snapshot,
                    title=signal,
                    kind=kind,
                    semantic=preview.semantic,
                    overlay_signal=overlay_signal,
                    initial_publication=publication,
                )
            except Exception as error:
                error_key = f"{signal}|{type(error).__name__}:{error}"
                errors = self._preview_errors.setdefault(binding.node_id, set())
                if error_key not in errors:
                    errors.add(error_key)
                    self._report(
                        f"{binding.node_id}: cannot open preview {signal!r}: "
                        f"{_error_text(error)}",
                        severity="error",
                    )
                continue
            binding.previewed += (signal,)
            if self._is_task(binding):
                self._auto_task_previews.setdefault(binding.node_id, {})[
                    panel.panel_id
                ] = signal

    def _reconcile_task_previews(self, binding: LogicBinding) -> None:
        tracked = self._auto_task_previews.pop(binding.node_id, {})
        for panel_id in tracked:
            self._remove_panel_now(panel_id)

    def _operator_point_review(
        self, binding: LogicBinding, request: OperatorInputRequest
    ) -> None:
        host = binding.host
        if host is None:
            return
        output_name = str(request.payload.get("output_name", "")).strip()
        producer = str(request.payload.get("producer", "")).strip()
        owner = binding.node_id if not producer else f"{binding.node_id}/{producer}"
        signal = stable_signal_key(owner, output_name)
        front = self.session.signal_plane.freeze()
        publication = front.publication(signal)
        value = front.value(signal)
        if publication is None or value is None:
            raise RuntimeError("operator point review has no published image")
        snapshot, _event_record = self._presentation_snapshot(
            signal,
            value,
            publication,
        )
        geometry = publication.run_record.get(
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD
        )
        if not isinstance(geometry, Mapping):
            raise RuntimeError("operator point review image has no point geometry")
        point_ids = tuple(str(value) for value in geometry["point_ids"])
        requested_ids = tuple(
            str(value) for value in request.payload.get("point_ids", ())
        )
        if requested_ids != point_ids:
            raise RuntimeError("operator point review identities differ from the image")
        overlay = ImagePointOverlay(
            revision=1,
            coordinates=geometry["coordinates_xy"],
            point_ids=point_ids,
            labels=tuple(str(value) for value in geometry["labels"]),
            static_statuses=tuple(PointStatus.UNKNOWN for _ in point_ids),
        )
        state = PanelState(
            signal=signal,
            kind=PlotKind.IMAGE.value,
            cell_kind="",
            size="4x4",
            interval_ms=self._default_interval_ms,
            title=request.title,
        )
        review_host = self._make_host(ImageFrame(snapshot, overlay), state)
        try:
            reviewer = self._review_points
            if reviewer is None:
                raise RuntimeError("TaskConsole has no point-review UI")
            excluded = reviewer(review_host, overlay, request)
            if excluded is None:
                host.cancel("operator cancelled site review")
            else:
                host.submit_operator_input(
                    request.request_id,
                    {"excluded_point_ids": tuple(str(value) for value in excluded)},
                )
        finally:
            self._retire_plot_host(review_host)

    def _handle_operator_request(self, binding: LogicBinding) -> None:
        host = binding.host
        request = None if host is None else host.operator_request
        if request is None:
            binding.operator_request_id = ""
            return
        if binding.operator_request_id == request.request_id:
            return
        binding.operator_request_id = request.request_id
        try:
            if request.kind != "point-selection":
                raise RuntimeError(
                    f"TaskConsole does not support operator request {request.kind!r}"
                )
            self._operator_point_review(binding, request)
        except BaseException as error:
            if host is not None and host.running:
                host.cancel(f"operator input failed: {_error_text(error)}")
            self._report(
                f"{binding.node_id}: operator input failed: {_error_text(error)}",
                severity="error",
            )

    def _finish_task_takeover(
        self,
        binding: LogicBinding,
        *,
        status: str,
        severity: str,
    ) -> None:
        if self._active_task_id != binding.node_id:
            return
        self._reconcile_task_previews(binding)
        self._active_task_id = None
        self._project_task_takeover()
        self._report(f"{binding.node_id}: {status}", severity=severity)

    def _sync_task_takeover(self) -> None:
        binding = self._active_task()
        if binding is None:
            if self._active_task_id is not None:
                self._active_task_id = None
                self._project_task_takeover()
            return
        if binding.pending is not None:
            self._project_task_takeover()
            return
        host = binding.host
        if host is not None and host.running:
            self._project_task_takeover()
            return
        _state, status = self._logic_state(binding)
        error = "" if host is None else str(host.observation.error or "")
        if binding.draft_error:
            error = binding.draft_error
        self._finish_task_takeover(
            binding,
            status=error or status,
            severity="error" if error else "task",
        )

    def logic_offer(self) -> tuple[tuple[str, str, str, str], ...]:
        """Every addable row type without resolving or building a run."""

        return tuple(
            (api_name, kind, publishes, "")
            for api_name, kind, publishes in self.catalog.rows()
        )

    def installation_changed(self) -> None:
        """Re-project stopped drafts against the current live device set."""

        for binding in self.logic.values():
            binding.finalization_key = ()
            binding.finalization = None
            self.refresh_logic_editor(binding.node_id)
        self._refresh_console_projection()

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

        if self._task_command_blocked("adding a Logic Node"):
            return ""

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
        self.view.add_logic_row(
            selected_id, kind, descriptor.offers_a_preview
        )
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
        if open_editor:
            self._open_logic_editor(binding)
        self._refresh_console_projection()
        self._report(f"added {selected_id}", severity="task")
        return selected_id

    def _logic_finalization_key(self, binding: LogicBinding) -> tuple:
        """In-memory revisions that may change one draft's admission."""

        source_options = self._source_options(binding.descriptor, binding.node_id)
        source = binding.draft.source_signal.strip()
        publication = (
            self.session.signal_plane.latest_publication(source)
            if source
            else None
        )
        armed = bool(source) and self.session.signal_plane.is_generation_live(
            source
        )
        return (
            binding.draft_revision,
            id(self.session.installation),
            int(getattr(self.session.installation, "revision", 0)),
            source_options,
            publication is not None,
            armed,
        )

    def _finalize_logic_binding(
        self,
        binding: LogicBinding,
        *,
        force: bool = False,
    ) -> LogicDraftFinalization:
        """Cache one owner finalization until its raw or external facts change."""

        key = self._logic_finalization_key(binding)
        if force or binding.finalization is None or binding.finalization_key != key:
            binding.finalization = finalize_logic_draft(
                binding.descriptor,
                binding.draft,
                installation=self.session.installation,
                signal_plane=self.session.signal_plane,
                workspace=self.session.workspace,
                source_options=self._source_options(
                    binding.descriptor, binding.node_id
                ),
            )
            binding.finalization_key = key
        return binding.finalization

    def logic_editor_projection(self, node_id: str) -> dict[str, Any] | None:
        """Plain state consumed by Logic Edit and future producer projections."""

        binding = self.logic.get(str(node_id))
        if binding is None:
            return None
        from .authoring_form import (
            display_value,
            project_artifact_inputs,
            project_logic_schema,
        )

        finalization = self._finalize_logic_binding(binding)
        options = device_key_options(
            binding.descriptor,
            installation=self.session.installation,
        )
        artifact_specs = artifact_input_specs(binding.descriptor)
        workspace = getattr(self.session, "workspace", None)
        artifact_base_dir = str(getattr(workspace, "data", ""))
        state, status = self._logic_state(binding)
        resource_fields = {
            spec.field_name for spec in binding.descriptor.workspace_resources
        }
        resource_directories = {
            spec.field_name: (
                Path(self.session.workspace.root).resolve() / spec.directory
            ).resolve()
            for spec in binding.descriptor.workspace_resources
        }
        form_values = {}
        for field in binding.descriptor.authoring_schema.fields:
            value = (
                finalization.values[field.name]
                if field.name in finalization.field_availability
                and field.name in finalization.values
                else binding.draft.values.get(field.name, field.default)
            )
            if field.name in resource_fields and value:
                selected = Path(str(value)).expanduser()
                if not selected.is_absolute():
                    selected = resource_directories[field.name] / selected
                value = str(selected.resolve())
            form_values[field.name] = display_value(value)
        can_start = finalization.can_start and binding.pending is None
        can_stop = bool(
            binding.pending is not None
            or (binding.host is not None and binding.host.running)
        )
        source_specs = dataset_inputs(binding.descriptor)
        return {
            "node_id": binding.node_id,
            "api_name": str(binding.descriptor.api_name),
            "kind": str(
                getattr(binding.descriptor.kind, "value", binding.descriptor.kind)
            ),
            "form_spec": project_logic_schema(
                binding.descriptor,
                workspace_root=str(self.session.workspace.root),
                field_availability=finalization.field_availability,
            ),
            "form_values": form_values,
            "artifact_form_spec": project_artifact_inputs(
                artifact_specs,
                base_dir=artifact_base_dir,
            ),
            "artifact_values": dict(binding.draft.artifact_inputs),
            "artifact_results": self._artifact_results(binding),
            # Beside Start, in both places an operator can press it.  One
            # preference, projected twice; neither widget keeps a default --
            # and no switch at all where the node opens nothing.
            "auto_preview": binding.auto_preview,
            "preview_offered": binding.descriptor.offers_a_preview,
            "source_required": bool(dataset_inputs(binding.descriptor)),
            "source_label": (
                source_specs[0].name.replace("_", " ").title()
                if source_specs
                else "Signal"
            ),
            "source_signal": binding.draft.source_signal,
            "source_options": self._source_options(
                binding.descriptor, binding.node_id
            ),
            "source_labels": self._source_labels(
                binding.descriptor, binding.node_id
            ),
            "source_groups": self._source_groups(
                binding.descriptor, binding.node_id
            ),
            "device_keys": dict(binding.draft.device_keys),
            "device_options": options,
            "ui_contributions": tuple(binding.descriptor.ui_contributions),
            "workspace_resources": dict(finalization.resources),
            "running": bool(binding.host is not None and binding.host.running),
            "pending": binding.pending is not None,
            "can_start": can_start,
            "can_stop": can_stop,
            "issues": finalization.issues,
            "error": status if state == "error" else "",
            "status": status,
            # The offer-relevant subset of what a start would bind: editors
            # project from these without the start-time side effects.
            "bench_extras": self._bench_offer_extras(),
        }

    def _open_logic_editor(self, binding: LogicBinding) -> bool:
        # Opening/focusing edit is the explicit resource refresh boundary.
        # The heartbeat never polls the filesystem.
        self._finalize_logic_binding(binding, force=True)
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

        if self._task_command_blocked(
            "changing a logic draft", node_id=str(node_id)
        ):
            return False

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
        binding.draft_revision += 1
        binding.finalization_key = ()
        binding.finalization = None
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
        if self._task_command_blocked("starting another logic node"):
            return False
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        finalization = self._finalize_logic_binding(binding, force=True)
        if not finalization.can_start:
            error = finalization.issues[0]
            binding.draft_error = error
            self._report(f"{node_id}: {error}", severity="error")
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            return False
        if self._processor_source_absent(binding, finalization):
            # A processor is a standing follower, not a one-shot run: with
            # no source signal yet there is nothing to bind to, so the Start
            # is accepted as an intent and the poll beat completes it the
            # moment the source publishes.
            binding.following = True
            binding.draft_error = ""
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            self._report(
                f"{node_id} following {finalization.source_signal}"
                " (waiting for the signal)",
                severity="task",
            )
            return True
        try:
            candidate = self._build_logic_candidate(binding, finalization)
        except Exception as error:
            binding.draft_error = _error_text(error)
            self._report(f"{node_id}: {_error_text(error)}", severity="error")
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
            binding.draft_error = _error_text(error)
            self._report(f"{node_id}: {_error_text(error)}", severity="error")
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            return False
        except Exception as error:
            self._discard_candidate(binding, candidate)
            binding.draft_error = _error_text(error)
            self._report(f"{node_id}: {_error_text(error)}", severity="error")
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            return False

        blockers = set(candidate.reservation.waiting_for)
        if blockers:
            binding.pending = candidate
            self._begin_task_takeover(binding)
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            self._report(
                f"{node_id} queued while {', '.join(sorted(blockers))} stops",
                severity="task",
            )
            return True
        activated = self._activate_candidate(binding, candidate)
        if activated:
            if self._is_processor(binding):
                binding.following = True
            self._begin_task_takeover(binding)
            self._refresh_console_projection()
        return activated

    @staticmethod
    def _is_processor(binding: LogicBinding) -> bool:
        kind = getattr(binding.descriptor, "kind", None)
        return getattr(kind, "value", kind) == "processor"

    def _processor_source_absent(
        self, binding: LogicBinding, finalization: object
    ) -> bool:
        if not self._is_processor(binding):
            return False
        signal = str(getattr(finalization, "source_signal", "") or "")
        if not signal:
            return False
        plane = self.session.signal_plane
        # An ARMED silent source is attachable now; only a source that is
        # neither publishing nor armed leaves the processor following.
        return plane.latest_publication(
            signal
        ) is None and not plane.is_generation_live(signal)

    def _follow_processor_sources(self) -> None:
        """Complete standing processor Starts whose source is alive again."""

        if self._active_task() is not None:
            # A Task owns the bench; deferring the follow to a later beat is
            # waiting, not giving up -- and start_logic would report the
            # block on every beat.
            return
        for binding in tuple(self.logic.values()):
            if (
                not binding.following
                or binding.removing
                or binding.pending is not None
            ):
                continue
            host = binding.host
            if host is not None and (
                host.running
                or host.observation.phase not in ("done", "cancelled")
            ):
                # Running follows by itself; a failure is the operator's to
                # read, not this beat's to retry.  A CANCELLED host whose
                # following survived was cancelled by something other than
                # the operator (their Stop clears the flag) -- a device
                # takeover, say -- and follows again like a finished one.
                if host.observation.phase == "failed":
                    binding.following = False
                continue
            finalization = binding.finalization
            signal = str(getattr(finalization, "source_signal", "") or "")
            if not signal:
                continue
            plane = self.session.signal_plane
            if not plane.is_generation_live(signal):
                # Not armed, or the retained tail of a finished run: a
                # frozen pass already answered the latter, and re-running
                # forever would spin.  An armed source needs no publication
                # to be followed -- the follower's own start may be what
                # causes the first one.
                continue
            if not self.start_logic(binding.node_id):
                # A start the source's own lifecycle refused -- it ended or
                # moved on between this beat's gate and the bind, which the
                # host reports by ending CANCELLED -- is the exact race
                # this follower exists for: keep following and let a later
                # beat complete it.  A start refused with the source alive
                # and the host not cancelled is structural, and the
                # operator's to read.
                host = binding.host
                lifecycle = (
                    host is not None
                    and host.observation.phase == "cancelled"
                )
                if not lifecycle and plane.is_generation_live(signal):
                    binding.following = False

    def stop_logic(self, node_id: str) -> bool:
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        binding.following = False
        had_pending = binding.pending is not None
        self._discard_pending(binding)
        host = binding.host
        if host is not None and host.running:
            host.cancel("the operator pressed Stop")
        if self._active_task_id == binding.node_id and (
            host is None or not host.running
        ):
            _state, status = self._logic_state(binding)
            self._finish_task_takeover(
                binding,
                status="cancelled" if had_pending else status,
                severity="task",
            )
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

        if self._task_command_blocked(
            "removing a logic node", node_id=str(node_id)
        ):
            return False

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
                self._report(
                    f"{binding.node_id}: {_error_text(error)}", severity="error"
                )
                return False
        if binding.lease is not None:
            binding.lease.release()
            binding.lease = None
        self.logic.pop(binding.node_id, None)
        self._preview_errors.pop(binding.node_id, None)
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
                    # Stop owns the Task lifecycle.  A failed or abandoned
                    # Plot surface must never prevent Runtime from accepting
                    # the worker's terminal completion; cancelled partial
                    # publications are sealed by NodeHost before this owner
                    # releases the device lease.
                    binding.host.poll()
                except Exception as error:
                    self._report(
                        f"{binding.node_id}: {_error_text(error)}",
                        severity="error",
                    )
                if not self._closing:
                    self._capture_artifact_results(binding)
                    self._ensure_node_previews(binding)
                    self._handle_operator_request(binding)
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
        self._follow_processor_sources()
        self._sync_task_takeover()
        self._refresh_console_projection()

    def _generation_surface_busy(self, host: object) -> bool:
        """Whether a Panel is still consuming this Host's causal generation."""

        generation = host.generation
        if generation is None:
            return False
        owner = host.instance_id
        for panel in self.panels.values():
            port = panel.port
            if port is None or not port.surface_busy:
                continue
            for signal in port.front_signals:
                publication = self.session.signal_plane.latest_publication(signal)
                if publication is None:
                    continue
                roots = self.session.signal_plane.publication_roots(publication)
                if any(
                    root.stream_id.value == owner
                    and root.generation == generation
                    for root in roots
                ):
                    return True
        return False

    @staticmethod
    def _observation_status(observed: object) -> str:
        phase = str(getattr(observed, "phase", "") or "running")
        if bool(getattr(observed, "terminal", False)) or phase == "stopping":
            return phase
        progress = getattr(observed, "progress", None)
        text = str(getattr(progress, "text", "") or "")
        if text:
            return text
        return phase

    def _logic_state(self, binding: LogicBinding) -> tuple[str, str]:
        host = binding.host
        if host is None:
            issues = self._finalize_logic_binding(binding).issues
            error = binding.draft_error or (issues[0] if issues else "")
            state = "error" if error else "idle"
            status = error or "not started"
            if binding.following and not error:
                signal = getattr(binding.finalization, "source_signal", "")
                state = "running"
                status = f"following {signal} (waiting for the signal)"
        else:
            observed = host.observation
            if observed.error:
                state, status = "error", observed.error
            elif observed.running:
                state, status = "running", self._observation_status(observed)
            elif binding.draft_error:
                state, status = "error", binding.draft_error
            else:
                issues = self._finalize_logic_binding(binding).issues
                if issues:
                    state, status = "error", issues[0]
                elif binding.following:
                    signal = getattr(
                        binding.finalization, "source_signal", ""
                    )
                    state = "running"
                    status = f"following {signal} (source stopped)"
                else:
                    state, status = "idle", self._observation_status(observed)
            warnings = tuple(getattr(observed, "warnings", ()))
            if warnings:
                status = f"{status}; warning: {'; '.join(warnings)}"
        if binding.pending is not None:
            waiting = ", ".join(sorted(binding.pending.waiting_for))
            state = "running"
            status = f"waiting for {waiting}" if waiting else "restart queued"
        return state, str(status)

    def _show_logic(self, binding: LogicBinding) -> None:
        """What one node is doing, pushed only when it changed.

        A row rewritten every beat is a row an operator cannot read a status
        off, because the text they were halfway through replaced itself.
        """

        host = binding.host
        finalization = self._finalize_logic_binding(binding)
        state, status = self._logic_state(binding)
        direct_names = (
            host.published_signals()
            if host is not None
            else tuple(
                stable_signal_key(binding.node_id, output.name)
                for output in self._logic_outputs(binding)
            )
        )
        descriptions = {
            item.name: item for item in self.session.signal_plane.describe_signals()
        }
        names = tuple(dict.fromkeys(direct_names))
        published = []
        for name in names:
            description = descriptions.get(name)
            if description is None or description.shape is None:
                lifecycle = "waiting"
            else:
                lifecycle = "live" if description.live else "finished"
            published.append(
                (
                    name.rsplit("/", 1)[-1] or name,
                    format_signal_shape(
                        None if description is None else description.shape
                    ),
                    f"{lifecycle} · {name}",
                )
            )
        published = tuple(published)
        artifacts = self._artifact_results(binding)
        can_start = finalization.can_start and binding.pending is None
        can_stop = bool(
            binding.pending is not None or (host is not None and host.running)
        )
        shown = (
            state,
            status,
            published,
            artifacts,
            can_start,
            can_stop,
            binding.auto_preview,
        )
        if shown == binding.shown:
            return
        binding.shown = shown
        self.view.set_logic_state(binding.node_id, state, status)
        self.view.set_logic_commands(
            binding.node_id,
            can_start=can_start,
            can_stop=can_stop,
        )
        self.view.set_logic_publishes(binding.node_id, published)
        self.view.set_logic_auto_preview(binding.node_id, binding.auto_preview)
        self.refresh_logic_editor(binding.node_id)

    def _source_options(
        self,
        descriptor: Any,
        consumer_node_id: str,
    ) -> tuple[str, ...]:
        """Stable keys whose declared Dataset contract matches this input."""

        specs = dataset_inputs(descriptor)
        if not specs:
            return ()

        def accepts(contract_id: str | None) -> bool:
            return any(spec.accepts(contract_id) for spec in specs)

        compatible: set[str] = set()
        for binding in self.logic.values():
            if binding.node_id == consumer_node_id:
                continue
            for output in self._logic_outputs(binding):
                if accepts(str(output.contract_id)):
                    compatible.add(stable_signal_key(binding.node_id, output.name))
        compatible.update(
            description.name
            for description in self.session.signal_plane.describe_signals()
            if description.owner_id != consumer_node_id
            and accepts(description.contract_id)
        )
        return tuple(sorted(compatible))

    def _source_labels(
        self,
        descriptor: Any,
        consumer_node_id: str,
    ) -> dict[str, str]:
        compatible = set(
            self._source_options(descriptor, consumer_node_id)
        )
        return {
            row.name: row.label
            for row in project_signals(self.session.signal_plane)
            if row.name in compatible
        }

    def _source_groups(
        self,
        descriptor: Any,
        consumer_node_id: str,
    ) -> dict[str, str]:
        """Which producer each compatible source belongs under.

        The same producer grouping every signal chooser shows: the plane's
        projection for published signals, and the declaring node's id for a
        compatible output that has not published yet.
        """

        compatible = set(self._source_options(descriptor, consumer_node_id))
        groups = {
            row.name: row.producer
            for row in project_signals(self.session.signal_plane)
            if row.name in compatible
        }
        for binding in self.logic.values():
            for output in self._logic_outputs(binding):
                key = stable_signal_key(binding.node_id, output.name)
                if key in compatible:
                    groups.setdefault(key, binding.node_id)
        return groups

    def _build_logic_candidate(
        self,
        binding: LogicBinding,
        finalization: LogicDraftFinalization,
    ) -> LogicCandidate:
        """Freeze and build one complete candidate without touching old runs."""

        arguments = build_arguments(
            binding.descriptor,
            signal_plane=self.session.signal_plane,
            finalization=finalization,
            extras=self._logic_extras(),
        )
        node = binding.descriptor.instantiate(**arguments)
        previews = tuple(binding.descriptor.node_previews)
        host = make_host(
            binding.descriptor,
            node,
            signal_plane=self.session.signal_plane,
            instance_id=binding.node_id,
            source_signal=finalization.source_signal or None,
            request_owner_wake=self.board.wake.request_owner_wake,
        )
        claims = tuple(
            DeviceClaim(
                requirement.argument_name,
                finalization.device_keys[requirement.argument_name],
                arguments[requirement.argument_name],
                requirement.protected_fields,
            )
            for requirement in binding.descriptor.device_requirements
        )
        resolve_claims = getattr(node, "resolved_device_claims", None)
        if callable(resolve_claims):
            from zlc_atom.nodes import ResolvedDeviceClaim

            runtime_claims = tuple(resolve_claims())
            if any(
                not isinstance(claim, ResolvedDeviceClaim)
                for claim in runtime_claims
            ):
                raise TypeError(
                    "resolved_device_claims must contain ResolvedDeviceClaim values"
                )
            claims += tuple(
                DeviceClaim(
                    f"runtime:{claim.device_key}",
                    claim.device_key,
                    claim.device,
                    claim.protected_fields,
                    False,
                )
                for claim in runtime_claims
            )
        is_task = str(
            getattr(binding.descriptor.kind, "value", binding.descriptor.kind)
        ) == "task"
        return LogicCandidate(
            node,
            host,
            previews,
            claims,
            run_root=self.session.day_folder() if is_task else None,
            input_summary=(
                task_input_summary(binding.descriptor, finalization)
                if is_task
                else {}
            ),
        )

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
            self._report(f"{binding.node_id}: {_error_text(error)}", severity="error")

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
            # Beginning the replacement generation retires this run and its
            # whole derived closure from the Plane.  A Panel projection that
            # already reserved the old publication still has to materialize
            # it there, so let only those causally-related surfaces finish
            # before withdrawing the generation.  Manual Stop then Start had
            # this drain interval naturally; Restart must provide the same
            # lifecycle boundary without blanking or rebuilding the Panel.
            if self._generation_surface_busy(old_host):
                binding.pending = candidate
                self._refresh_console_projection()
                return True
            try:
                old_host.shutdown()
            except Exception as error:
                self._discard_candidate(binding, candidate)
                binding.draft_error = _error_text(error)
                self._report(
                    f"{binding.node_id}: {_error_text(error)}", severity="error"
                )
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
            binding.draft_error = _error_text(error)
            self._report(f"{binding.node_id}: {_error_text(error)}", severity="error")
            self._refresh_console_projection()
            return False
        candidate.reservation = None
        binding.node = candidate.node
        binding.host = candidate.host
        binding.preview_specs = candidate.previews
        binding.lease = lease
        binding.pending = None
        binding.artifact_results = ()
        binding.artifact_result_host = None
        binding.artifact_completion_order = 0
        try:
            if candidate.run_root is None:
                candidate.host.start()
            else:
                candidate.host.start(
                    run_root=candidate.run_root,
                    input_summary=candidate.input_summary,
                )
        except Exception as error:
            lease.release()
            binding.lease = None
            binding.draft_error = _error_text(error)
            self._report(f"{binding.node_id}: {_error_text(error)}", severity="error")
            self._refresh_console_projection()
            return False
        binding.draft_error = ""
        self._refresh_console_projection()
        self._report(f"{binding.node_id} started", severity="task")
        return True

    def _fitting_cell_kind(self, subject: object, kind: str, declared: str) -> str | None:
        """The cell kind this panel will draw with, or None as an honest refusal.

        An empty declaration means the DATA decides -- the resolver
        (``fitting_panel_spec``) owns that rule and keeps applying it on every
        later retarget.  A named cell kind is the operator's choice: probe it,
        never swap it.  Refusing loudly beats the blank card this replaces.
        """

        spec = (
            self._spec_for_value(subject, kind, declared)
            if hasattr(subject, "snapshot")
            else self._spec_for(subject, kind, declared)
        )
        if spec is None:
            return None
        if kind != "facet_grid":
            return declared
        return semantic_spec(spec).kind.value

    def _spec_for_value(
        self,
        value: object,
        kind: str,
        cell_kind: str = "",
    ) -> Any:
        """Resolve plot compatibility from canonical geometry without values."""

        schema = getattr(value, "canonical_schema", None)
        if schema is not None:
            try:
                return task_console_fitting_spec(schema, kind, cell_kind)
            except Exception:
                return None
        snapshot = getattr(value, "snapshot", None)
        return None if snapshot is None else self._spec_for(snapshot, kind, cell_kind)

    def _spec_for(self, snapshot: object, kind: str, cell_kind: str = "") -> Any:
        """Whether this data can be drawn as ``kind``, as the plotting package sees it.

        A probe, not a guess: the same call that builds the spec answers it, so
        offering a kind and building it cannot disagree.
        """

        if self._spec_probe is None:
            return object()
        try:
            return self._spec_probe(snapshot, kind, cell_kind)
        except Exception:
            return None

    def _bench_offer_extras(self) -> dict[str, Any]:
        """Bench facts an EDITOR may offer from: side-effect free by contract.

        Run paths are deliberately absent: opening an editor must not create
        the day folder or otherwise touch the filesystem.
        """

        from zlc_atom.install import tunable_devices

        return {"tunable_devices": tunable_devices(self.session.installation)}

    def _logic_extras(self) -> dict[str, Any]:
        """Facts a START can bind beyond its devices and the signal plane."""

        return self._bench_offer_extras()

    def _artifact_results(
        self,
        binding: LogicBinding,
    ) -> tuple[Mapping[str, str], ...]:
        """Already-observed saved paths from one successful current host."""

        if binding.artifact_result_host is binding.host:
            return binding.artifact_results
        return ()

    def _capture_artifact_results(self, binding: LogicBinding) -> None:
        """Project this Task run directory and its explicitly registered files."""

        host = binding.host
        run_directory = None if host is None else getattr(host, "run_directory", None)
        if host is None or run_directory is None:
            return
        rows: list[Mapping[str, str]] = []
        for artifact in getattr(host, "artifacts", ()):
            rows.append(
                {
                    "name": artifact.name,
                    "contract_id": artifact.contract_id,
                    "path": str(artifact.path),
                    "role": artifact.role,
                }
            )
        rows.append(
            {
                "name": "run_directory",
                "contract_id": "zlc.task-run",
                "path": str(run_directory),
                "role": "run",
            }
        )
        binding.artifact_result_host = host
        binding.artifact_results = tuple(rows)
        if (
            host.terminal
            and host.observation.error is None
            and binding.artifact_completion_order == 0
            and any(row.get("role") == "final" for row in rows)
        ):
            self._artifact_completion_order += 1
            binding.artifact_completion_order = self._artifact_completion_order

    def _default_artifact_inputs(self, descriptor: object) -> dict[str, str]:
        """Freeze the latest observed matching artifact into a new row draft."""

        available: dict[str, tuple[int, str]] = {}
        for binding in self.logic.values():
            if binding.artifact_completion_order <= 0:
                continue
            for row in self._artifact_results(binding):
                if row.get("role") != "final" or not row.get("contract_id"):
                    continue
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
        self._project_task_takeover()
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

    def _begin_close(self) -> None:
        self._closing = True
        self._close_started_at = time.monotonic()
        self._report("closing console; waiting for active work", severity="task")
        for binding in tuple(self.logic.values()):
            binding.removing = True
            self._discard_pending(binding)
            if binding.host is not None and binding.host.running:
                binding.host.cancel("the console is closing")
        for panel_id in list(self.panels):
            self._remove_panel_now(panel_id)
        self.board.close()

    def _advance_close(self) -> bool:
        board_closed = self.board.close()
        owners_ready = (
            not self.logic
            and not self._retired_plot_hosts
            and board_closed
            and not self._saving_panels
        )
        worker_closed = self._close_worker() if owners_ready else False
        ready = owners_ready and worker_closed
        if ready:
            self._closed = True
            if self._request_close is not None and not self._close_retry_sent:
                self._close_retry_sent = True
                self._request_close()
            return True
        started = self._close_started_at
        if (
            started is not None
            and not self._close_wait_reported
            and time.monotonic() - started >= self.CLOSE_REPORT_SECONDS
        ):
            waiting = [binding.node_id for binding in self.logic.values()]
            if self.board.pending_projection_count:
                waiting.append(
                    f"{self.board.pending_projection_count} panel projection(s)"
                )
            if self._retired_plot_hosts:
                waiting.append(f"{len(self._retired_plot_hosts)} plot worker(s)")
            if self._saving_panels:
                waiting.append(f"{len(self._saving_panels)} panel save(s)")
            elif owners_ready and not worker_closed:
                waiting.append("panel-save worker")
            self._report(
                "console close is still waiting for " + ", ".join(waiting),
                severity="error",
            )
            self._close_wait_reported = True
        return False

    def close(self) -> bool:
        """Close guard: initiate shutdown once and never wait on the Qt owner."""

        if self._closed:
            return True
        first = not self._closing
        if first:
            self._begin_close()
        ready = self._advance_close()
        return False if first else ready
