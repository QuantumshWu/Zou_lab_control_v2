"""Start the task console.

    zlc task_console --workspace D:/experiment

This is the composition root at its most literal: it builds the session, the
views, the presenter and the display beat, connects them, and gets out of the
way.  Every decision it appears to make is really a default being passed
through -- which apparatus, which pulse, which signal to show first.

The window drives the same ExperimentSession a notebook drives.  If a button
ever needs something the notebook cannot do, that capability is missing from the
session, not from the window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


import logging

_LOG = logging.getLogger(__name__)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the neutral-atom task console.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "directory holding pulses/, data/ and apparatus.json "
            "(default: found at or above this one)"
        ),
    )
    parser.add_argument(
        "--template",
        default=None,
        help="start from a named apparatus template instead of apparatus.json (e.g. virtual)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build everything and exit without opening a window (a startup smoke test)",
    )
    return parser


def _beat_interval_ms(presenter) -> int:
    """Poll at the display clock's base; overdue deadlines use monotonic time."""

    return int(presenter.board.base_interval_ms)


def open_experiment(workspace=None, template=None):
    """Open the shared Experiment session behind an initially empty console.

    Kept apart from the window so the same assembly serves the smoke check, a
    notebook, and the window entry -- three ways in, one experiment.
    """

    from ..session import ExperimentSession, Workspace

    space = Workspace(workspace) if workspace is not None else Workspace.discover()

    session = ExperimentSession.open(space.root, template=template)
    return space, session


def build_panel_host(
    plot_input,
    state,
    *,
    build_host,
    device_pixel_ratio: float = 1.0,
    initial_spec=None,
    initial_configuration=None,
):
    """One panel host from one panel state: THE mount path for every card.

    Module-level on purpose -- the presenter tests mount through this exact
    function, so a divergence between what the app builds and what the tests
    build cannot exist.

    The panel's stored appearance (``state.display``) is the complete current
    vocabulary.  During a kind/cell transition the mount takes only the legal
    intersection with the target vocabulary; the first successful description
    then replaces the transition bag wholesale.  Unknown names never reach a
    host and never survive as compatibility state.
    """

    from ..panel_catalog import task_console_fitting_spec
    from ..panel_state import project_panel_state

    if not callable(build_host):
        raise TypeError("build_host must be callable")

    snapshot = getattr(plot_input, "snapshot", plot_input)
    spec = initial_spec if initial_spec is not None else task_console_fitting_spec(
        snapshot.block.schema, state.kind, state.cell_kind)
    if spec is None:
        raise ValueError(
            f"{state.signal!r} cannot be drawn as {state.kind or 'anything'}"
        )
    projection = project_panel_state(snapshot.block.schema, spec, state)
    if not projection.drawable:
        # A figure host draws; a table with a vacant required role does not.
        raise ValueError(
            f"{state.signal!r} cannot be drawn: {projection.vacancy}"
        )
    spec, parameters = projection.spec, projection.parameters
    return build_host(
        plot_input,
        spec,
        size=state.size,
        parameters=parameters,
        device_pixel_ratio=device_pixel_ratio,
        initial_configuration=initial_configuration,
    )


def panel_host_factories(view, monitor_render, editor_render):
    """One window's Monitor and Edit/Save host factories.

    Both mount through :func:`build_panel_host` at the window's CURRENT
    screen scale, read per host: the Monitor factory runs on the projection
    lane, so it must not touch Qt, and it must not keep the scale of the
    screen the window first opened on.
    """

    def monitor(plot_input, state, **initial):
        return build_panel_host(
            plot_input,
            state,
            build_host=monitor_render.build_host,
            device_pixel_ratio=float(view.device_pixel_ratio()),
            **initial,
        )

    def editor(plot_input, state):
        return build_panel_host(
            plot_input,
            state,
            build_host=editor_render.build_host,
            device_pixel_ratio=float(view.device_pixel_ratio()),
        )

    return monitor, editor


def render_processes_closer(monitor_render, editor_render):
    """Let go of one window's claim on the Monitor pool and Edit/Save process.

    The returned call never blocks by default and may be repeated: the first
    releases both claims, and each answers True once every service this
    window was the last to hold has closed.  A ``timeout`` waits that long
    for them instead -- the abandon path of a window that failed to open.
    """

    released = False
    monitor_shutdown = editor_shutdown = False

    def close(timeout: float = 0.0) -> bool:
        nonlocal released, monitor_shutdown, editor_shutdown
        if not released:
            monitor_shutdown = not monitor_render.release(timeout=0.0)
            editor_shutdown = not editor_render.release(timeout=0.0)
            released = True
        monitor_closed = (
            monitor_render.close(timeout=timeout) if monitor_shutdown else True
        )
        editor_closed = (
            editor_render.close(timeout=timeout) if editor_shutdown else True
        )
        return bool(monitor_closed and editor_closed)

    return close


def staged_panel_surface(host):
    """A board panel's widget STAGES its fronts; the board presents them.

    Auto-present would put each panel's pixels up the moment its own render
    lands, so two panels of one causal group (a camera frame and the
    occupancy derived from it) could show different shots.  With staging,
    the presenter presents each same-shot batch atomically and routes every
    non-batch render through the same present helpers.  Every window whose
    panels a ConsolePresenter drives -- the console and the figure viewer --
    mounts this one policy; a viewer panel is fed by the same Runtime/Panel
    derivations (manual Apply, ROI, Fit) as a console panel and is not
    exempt because its first Dataset came from an archive.
    """

    return host.qt_widget(auto_present=False)


def build_console(session, *, window_ratio=None, request_close=None, run_device_read=None):
    """One console presenter over one session, with the view it drives."""

    from ..panel_sizes import install as install_panel_sizes

    install_panel_sizes()
    import zlc_plot as plot

    from zlc_ui import open_task_console

    from ..board import attach_qt_owner_turn, attach_qt_worker
    from ..console import ConsolePresenter

    # One call, one handle: this layer never names a widget class.
    view = open_task_console(
        title="TaskConsole@Zou lab",
        window_ratio=window_ratio,
        plot_surface=staged_panel_surface,
    )
    monitor_render = plot.RenderProcessPool("zlc-monitor-render")
    try:
        editor_render = plot.RenderProcess("zlc-edit-save-render")
    except BaseException:
        monitor_render.release(timeout=30.0)
        view.close()
        raise

    close_renders = render_processes_closer(monitor_render, editor_render)
    close_device_read = None

    def _close_render_processes() -> bool:
        reads_closed = close_device_read is None or close_device_read()
        return bool(close_renders() and reads_closed)

    build_monitor_host, build_editor_host = panel_host_factories(
        view, monitor_render, editor_render
    )

    def _review_points(host, overlay, request):
        point_ids = tuple(overlay.point_ids or ())
        labels = tuple(
            point_id if label is None else str(label)
            for point_id, label in zip(
                point_ids,
                overlay.labels or (None,) * len(point_ids),
                strict=True,
            )
        )
        points = tuple(
            (point_id, label, float(coordinate[0]), float(coordinate[1]))
            for point_id, label, coordinate in zip(
                point_ids,
                labels,
                overlay.coordinates,
                strict=True,
            )
        )
        surface = plot.ImagePointReviewSurface(host, overlay)
        return view.review_points(
            surface,
            points,
            title=request.title,
            message=request.message,
            confirm_label=str(
                request.payload.get("confirm_label", "Continue")
            ),
            initial_excluded=tuple(
                request.payload.get("initial_excluded", ())
            ),
        )

    def _manual_axis(request):
        """Stand aside for the hand: the one thing no machine here does."""

        accepted = view.manual_axis_setting(
            title=request.title,
            message=request.message,
        )
        return {} if accepted else None

    try:
        if run_device_read is None:
            run_device_read, close_device_read = attach_qt_worker("zlc-console-device-read")
        presenter = ConsolePresenter(
            session,
            view,
            make_monitor_host=build_monitor_host,
            make_editor_host=build_editor_host,
            build_figure_host=editor_render.build_host,
            save_figure_artifact=editor_render.save_figure_artifact,
            close_render_processes=_close_render_processes,
            open_saved=lambda start: _open_saved_figure(
                view,
                start,
                workspace=session.workspace.root,
                monitor_render=monitor_render,
                editor_render=editor_render,
            ),
            request_close=view.close_later if request_close is None else request_close,
            review_points=_review_points,
            manual_axis=_manual_axis,
            run_device_read=run_device_read,
        )
    except BaseException:
        if close_device_read is not None:
            close_device_read()
        monitor_render.release(timeout=30.0)
        editor_render.release(timeout=30.0)
        view.close()
        raise
    # Completion-driven presentation: a finished render's wake hops to the GUI
    # thread and commits immediately, instead of waiting for the next beat.
    presenter.board.wake.set_notify(
        attach_qt_owner_turn(presenter.commit_surfaces)
    )
    # The window's Edit/Save render child, for the device editors opened
    # beside it: each holds its own claim, as a FigureViewer does.
    view.editor_render = editor_render
    return view, presenter


def _stands_outside(metadata, value: object) -> bool:
    """Whether a reading lies outside what the field may be commanded to.

    Only a number can: a switch has no window, and a field with no declared
    edge on a side cannot be outside it there.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    low, high = metadata.minimum, metadata.maximum
    return (low is not None and value < float(low)) or (
        high is not None and value > float(high)
    )


def _commandable(metadata, value: object) -> object:
    """The reading where it may be commanded, else the nearest bound.

    A Desired is a command the operator may send, and the field's bounds --
    the bench window narrowed by the instrument's own limits -- are what may
    be commanded.  An instrument found standing outside them (left there
    before the window was authored) is shown as it stands in Current; its
    Desired opens on the nearest value the bounds admit, because a Desired
    the bounds refuse is not a value the operator can do anything with, and
    a control that would not open at all took away the one place the knob
    could be moved back inside.
    """

    if not _stands_outside(metadata, value):
        return value
    low, high = metadata.minimum, metadata.maximum
    if low is not None and value < float(low):
        return type(value)(low) if isinstance(value, int) else float(low)
    return type(value)(high) if isinstance(value, int) else float(high)


class ExperimentGuiFlow:
    """One DeviceManager-owned session and its composition-owned windows."""

    def __init__(
        self,
        *,
        workspace=None,
        template=None,
        window_ratio=None,
    ) -> None:
        from ..session import Workspace

        self.space = Workspace(workspace) if workspace is not None else Workspace.discover()
        self.template = template
        self.catalog = None
        self.window_ratio = window_ratio
        self.devices = None
        self.session = None
        self.console = None
        self.console_presenter = None
        self.device_controls: dict[str, object] = {}
        self._device_control_devices: dict[str, object] = {}
        self.timer = None
        self._closing_console = False
        self._closing_all = False
        self._device_worker_run = None
        self._device_worker_close = None
        self._device_tune_active: str | None = None
        self._device_tune_pending: dict[tuple[str, str], object] = {}
        #: Everything one open control knows -- its reading, its drafts and
        #: the risk the operator accepted ("risk": (device session, owner
        #: revision) or None) -- keyed by device.
        self._device_control_models: dict[str, dict[str, object]] = {}
        self._device_refresh_active: set[str] = set()
        self._device_refresh_pending: set[str] = set()
        #: Devices whose control is being read for its first frame.
        self._device_control_opening: set[str] = set()
        self._device_shutdown_pending = False
        #: Queues ``_continue_device_shutdown`` onto the next owner turn.
        self._resume_shutdown = None

    def open(self) -> "ExperimentGuiFlow":
        from zlc_atom.install import (
            discover_device_catalog,
            installation_config_from_template,
        )

        from .device_manager import create_window as create_device_window
        from ..board import attach_qt_owner_turn, attach_qt_worker

        if self.devices is not None:
            return self
        catalog = discover_device_catalog()
        initial_config = (
            None
            if self.template is None
            else installation_config_from_template(catalog, str(self.template))
        )
        self.catalog = catalog
        # One flow-owned serial device worker: Manager discover/init/shutdown
        # and generic tune share its existing busy policy and one close truth.
        submit, close = attach_qt_worker("zlc-devices")
        continue_close = attach_qt_owner_turn(self.close)
        self._resume_shutdown = attach_qt_owner_turn(self._continue_device_shutdown)

        def run(work, deliver, failed):
            def finished(result):
                deliver(result)
                if self._closing_all:
                    continue_close()

            def failure(error):
                failed(error)
                # A failed shutdown stays visible; never replay device close.
                if self._closing_all and not self._device_shutdown_pending:
                    continue_close()

            submit(work, finished, failure)

        self._device_worker_run = run
        self._device_worker_close = close
        try:
            self.devices = create_device_window(
                workspace=self.space.root,
                window_ratio=self.window_ratio,
                catalog=catalog,
                initial_config=initial_config,
                initialize_session=self._initialize_session,
                on_initialized=self._open_work_windows,
                prepare_reconcile=self._prepare_session_reconcile,
                on_reconciled=self._session_reconciled,
                prepare_shutdown=self._prepare_session_shutdown,
                shutdown_session=self._shutdown_session,
                on_shutdown=self._session_shutdown_complete,
                on_device_open=self.open_device_control,
                on_device_published=lambda key: self._retire_device_controls(
                    frozenset({key})
                ),
                run_off_thread=run,
                close_worker=close,
            )
        except BaseException:
            close()
            self._device_worker_run = None
            self._device_worker_close = None
            raise
        self.devices.set_close_guard(self._device_manager_close_guard)
        return self

    def _initialize_session(self, config: object):
        from ..session import ExperimentSession

        if self.catalog is None:
            raise RuntimeError("experiment device catalog is not initialized")
        return ExperimentSession.from_config(self.space, config, catalog=self.catalog)

    def _open_work_windows(self, session: object) -> None:
        from ..board import attach_qt

        if self.session is not None:
            raise RuntimeError("this experiment flow already has an active session")
        console = presenter = timer = None
        try:
            console, presenter = build_console(
                session,
                window_ratio=self.window_ratio,
                request_close=self._console_owner_ready,
                run_device_read=self._device_worker_run,
            )
            timer = attach_qt(
                self._beat,
                interval_ms=_beat_interval_ms(presenter),
                board=presenter.board,
            )
            console.presenter = presenter
            console.session = session
            console.set_close_guard(self._console_close_guard)
        except BaseException:
            if timer is not None:
                timer.stop()
            if presenter is not None:
                presenter.close()
            if console is not None:
                console.set_close_guard(lambda: True)
                console.close()
            raise
        self.session = session
        self.console = console
        self.console_presenter = presenter
        self.timer = timer
        # The Device Manager stays.  It is not an installer that has served its
        # purpose: it is the bench window -- what is running, and the settings
        # those running devices accept -- so hiding it left an operator with no
        # way to reach a camera's gain without shutting the experiment down.
        # TaskConsole comes up in front; device controls remain on demand.
        self.devices.send_behind()

    def _beat(self) -> None:
        presenter = self.console_presenter
        if presenter is not None:
            presenter.beat()
        self._refresh_device_control_policies()
        if self._closing_all:
            self.close()

    def open_device_control(self, instance_id: str) -> object | None:
        """Open or raise the one control window for a loaded named device.

        A device with its own editor opens at once and is returned.  A
        generic control opens on the device's first reading -- the window
        is built from the fields the device declares, so it is read before
        there is a window to show -- and ``None`` is returned while that
        reading is under way.
        """

        key = str(instance_id)
        if (
            self.devices is not None
            and self.devices.presenter.device_operation_active
        ):
            raise RuntimeError("a Device Manager operation is still running")
        existing = self.device_controls.get(key)
        if existing is not None:
            hidden = not existing.is_visible()
            existing.restore()
            if hidden and key in self._device_control_models:
                self._project_device_control(key)
                self._request_device_control_refresh(key)
            return existing
        if key in self._device_control_opening:
            return None
        session = self.session
        if session is None or self.catalog is None:
            raise RuntimeError("initialize devices before opening a control")
        try:
            leaf = session.installation.devices[key]
        except KeyError as error:
            raise KeyError(f"no loaded device {key!r}") from error
        descriptors = {item.type_id: item for item in self.catalog.available}
        descriptor = descriptors[leaf.type_id]
        if descriptor.control_factory is None:
            self._open_generic_control(key, leaf.device)
            return None
        # A device editor's plots are Edit/export hosts: they draw in the
        # console's Edit/Save child, never here and never in a child of
        # their own.
        control = descriptor.control_factory(
            session,
            key,
            window_ratio=self.window_ratio,
            render=self.console.editor_render,
        )
        self._adopt_device_control(key, leaf.device, control)
        return control

    def _adopt_device_control(self, key: str, device: object, control: object) -> None:
        """This control is the one window for ``key`` until it closes."""

        control.set_device_label(self.session.device_labels.get(key, key))
        self.device_controls[key] = control
        self._device_control_devices[key] = device

        def released() -> None:
            if self.device_controls.get(key) is control:
                self._forget_device_control(key)
            if self._device_shutdown_pending and self._resume_shutdown is not None:
                # A shutdown waited for this control to close; it goes on
                # from the next owner turn, not from inside this close.
                self._resume_shutdown()

        control.closed.connect(released)

    def _forget_device_control(self, key: str) -> None:
        """Drop everything kept for one device's control, in one place."""

        self.device_controls.pop(key, None)
        self._device_control_devices.pop(key, None)
        self._device_control_models.pop(key, None)
        self._device_refresh_active.discard(key)
        self._device_refresh_pending.discard(key)
        for pending in tuple(self._device_tune_pending):
            if pending[0] == key:
                self._device_tune_pending.pop(pending, None)

    def _open_generic_control(self, key: str, device: object) -> None:
        """Read the device, then open its control on what it said.

        The window used to open at once, on an EMPTY spec with a placeholder
        note, and be re-projected once the worker had read the device -- so
        every control the operator opened painted a page with no rows, then
        rows with half their cells, then the page.
        The first frame is now the whole form: nothing is shown before the
        fields it is made of are known.  A second press while the reading
        is under way opens nothing; a reading that fails is reported where
        the press was made, on the Device Manager.
        """

        from zlc_ui import open_device_control

        session = self.session
        run = self._device_worker_run
        if session is None or run is None:
            raise RuntimeError("initialize devices before opening a control")
        if key in self._device_control_opening:
            return
        self._device_control_opening.add(key)

        def finish(result: object) -> None:
            self._device_control_opening.discard(key)
            if (
                self.session is not session
                or self._device_shutdown_pending
                or key in self.device_controls
            ):
                return
            try:
                model: dict[str, object] = {
                    "device": device,
                    "control": None,
                    "desired": {},
                    "live": {},
                    "status": {},
                    "device_session_id": "",
                    "risk": None,
                }
                self._adopt_device_reading(key, model, result)
                self._device_control_models[key] = model
                projection = self._device_control_projection(key)
                control = open_device_control(
                    title=f"{self.session.device_labels.get(key, key)} control",
                    spec=model["spec"],
                    projection=projection,
                )
                model["shown_owner_revision"] = projection["owner_revision"]
            except BaseException as error:
                self._device_control_models.pop(key, None)
                self._report_device(f"{key}: {error}", "error")
                return
            model["control"] = control
            # Each gesture guarded where it becomes a Qt slot: the projection
            # under these handlers can raise (a closing session, a device that
            # stopped answering), and an exception leaving a slot is qFatal --
            # the crash arrives ON the control the operator is touching, so the
            # report lands there too.
            control.refresh_requested.connect(
                self._guard_control_gesture(
                    control,
                    "refresh",
                    lambda selected=key: self._request_device_control_refresh(selected),
                )
            )
            control.risk_toggled.connect(
                self._guard_control_gesture(
                    control,
                    "risk toggle",
                    lambda accepted, selected=key: self._set_device_control_risk(
                        selected, accepted
                    ),
                )
            )
            control.field_desired_changed.connect(
                self._guard_control_gesture(
                    control,
                    "field edit",
                    lambda field, value, unit, selected=key: self._set_device_control_desired(
                        selected, field, value, unit
                    ),
                )
            )
            control.field_live_apply_toggled.connect(
                self._guard_control_gesture(
                    control,
                    "live toggle",
                    lambda field, enabled, selected=key: self._set_device_control_live(
                        selected, field, enabled
                    ),
                )
            )
            control.field_unit_requested.connect(
                self._guard_control_gesture(
                    control, "display unit",
                    lambda field, unit, selected=key: self._set_device_control_unit(selected, field, unit),
                )
            )
            control.field_apply_requested.connect(
                self._guard_control_gesture(
                    control,
                    "apply",
                    lambda field, value, unit, selected=key: self._queue_device_tune(
                        selected, field, value, unit
                    ),
                )
            )
            control.set_close_guard(
                lambda: self._generic_control_close_guard(key, control)
            )
            control.show_status(
                "ready" if model["tunables"] else "No runtime controls", "idle"
            )
            self._adopt_device_control(key, device, control)

        def failed(error: BaseException) -> None:
            self._device_control_opening.discard(key)
            self._report_device(f"{key}: {error}", "error")

        try:
            run(lambda: self._read_device_controls(device), finish, failed)
        except BaseException as error:
            failed(error)

    def _report_device(self, text: str, severity: str) -> None:
        """One line on the Device Manager, where the Control press was made."""

        if self.devices is not None:
            self.devices.show_status(text, severity)

    @staticmethod
    def _read_device_controls(
        device: object,
        units: dict[str, str] | None = None,
        *,
        refresh: bool = True,
    ) -> tuple[tuple[object, ...], dict[str, object], dict[str, object]]:
        """What the device declares and holds right now; runs on the worker."""

        from zlc_atom.authoring import (
            TunableField,
            is_tunable,
            read_tunable_in_unit,
            refresh_tunable_fields,
        )

        if not is_tunable(device):
            return (), {}, {}
        fields = refresh_tunable_fields(device) if refresh else tuple(device.tunable_fields())
        if any(not isinstance(field, TunableField) for field in fields):
            raise TypeError("device tunable_fields must contain TunableField values")
        if units:
            fields = tuple(
                read_tunable_in_unit(device, field.metadata.name, units[field.metadata.name])
                if units.get(field.metadata.name) and units[field.metadata.name] != field.metadata.unit
                else field for field in fields
            )
        current = {
            field.metadata.name: field.current for field in fields
        }
        provenance = {} if not fields else dict(device.settings_provenance())
        session_id = str(provenance.get("device_session_id", "")).strip()
        epoch = provenance.get("settings_epoch")
        if fields and (
            not session_id or type(epoch) is not int or epoch < 0
        ):
            raise ValueError("device settings provenance is invalid")
        return fields, current, provenance

    def _adopt_device_reading(
        self, key: str, model: dict[str, object], result: object
    ) -> None:
        """Put one reading of the device into its control model."""

        from zlc_atom.authoring import AuthoringSchema
        from ..authoring_form import project_schema

        fields, current, provenance = result
        names = tuple(field.metadata.name for field in fields)
        session_id = "" if not fields else str(provenance["device_session_id"])
        epoch = 0 if not fields else int(provenance["settings_epoch"])
        previous_session = str(model.get("device_session_id", ""))
        model["tunables"] = fields
        model["current"] = current
        model["spec"] = project_schema(
            AuthoringSchema(tuple(field.metadata for field in fields))
        )
        commandable = {
            field.metadata.name: (
                _commandable(field.metadata, current[field.metadata.name]), field.metadata.unit or ""
            )
            for field in fields
        }
        if previous_session != session_id:
            model["desired"] = commandable
            model["unit_drafts"] = set()
            model["unapplied"] = {}
            model["live"] = {name: False for name in names}
            model["risk"] = None
        else:
            desired = dict(model.get("desired", {}))
            model["desired"] = {
                name: desired.get(name, commandable[name]) for name in names
            }
            live = dict(model.get("live", {}))
            model["live"] = {
                name: bool(live.get(name, False)) for name in names
            }
        model["device_session_id"] = session_id
        model["settings_epoch"] = epoch
        model["status"] = {}

    def _request_device_control_refresh(self, key: str) -> None:
        key = str(key)
        model = self._device_control_models.get(key)
        run = self._device_worker_run
        if model is None or run is None:
            return
        if key in self._device_refresh_active:
            self._device_refresh_pending.add(key)
            return
        self._device_refresh_active.add(key)
        device = model["device"]
        control = model["control"]
        desired = dict(model["desired"])
        unit_requests = dict(model.get("unit_requests", {}))
        control.show_status("refreshing device settings", "task")

        def work():
            from zlc_atom.authoring import convert_tunable_value

            converted = {}
            for field, target in unit_requests.items():
                value, source = desired[field]
                converted[field] = (
                    None if value is None else convert_tunable_value(device, field, value, source, target),
                    target,
                )
            units = {name: converted.get(name, pair)[1] for name, pair in desired.items()}
            return self._read_device_controls(device, units, refresh=not unit_requests), converted

        def settled() -> None:
            self._device_refresh_active.discard(key)
            if key in self._device_refresh_pending:
                self._device_refresh_pending.discard(key)
                if key in self._device_control_models:
                    self._request_device_control_refresh(key)

        def finish(result: object) -> None:
            current_model = self._device_control_models.get(key)
            if current_model is not model or self.device_controls.get(str(key)) is not control:
                settled()
                return
            try:
                reading, converted = result
                if any(model["desired"].get(field) != desired[field]
                       or model.get("unit_requests", {}).get(field) != target
                       for field, target in unit_requests.items()):
                    self._device_refresh_pending.add(key)
                    return  # A newer edit gets its own complete read-only projection.
                self._adopt_device_reading(key, model, reading)
                for field, pair in converted.items():
                    model["desired"][field] = pair
                    model.setdefault("unit_drafts", set()).add(field)
                    model["unit_requests"].pop(field, None)
                self._project_device_control(key)
                control.show_status(
                    "ready" if model["tunables"] else "No runtime controls", "idle"
                )
            except BaseException as error:
                for field, target in unit_requests.items():
                    if model.get("unit_requests", {}).get(field) == target:
                        model["unit_requests"].pop(field, None)
                self._project_device_control(key)
                control.show_status(str(error), "error")
            finally:
                settled()

        def failed(error: BaseException) -> None:
            if self._device_control_models.get(key) is model:
                for field, target in unit_requests.items():
                    if model.get("unit_requests", {}).get(field) == target:
                        model["unit_requests"].pop(field, None)
                self._project_device_control(key)
                control.show_status(str(error), "error")
            settled()

        try:
            run(work, finish, failed)
        except BaseException as error:
            failed(error)

    def _device_control_projection(self, key: str) -> dict[str, object]:
        model = self._device_control_models[str(key)]
        session = self.session
        if session is None:
            raise RuntimeError("device session is closed")
        tunables = tuple(model.get("tunables", ()))
        names = tuple(field.metadata.name for field in tunables)
        groups = tuple(field.dependency_group for field in tunables)
        revision, owners, blockers = session.device_use.field_policy(
            key, names, dependency_groups=groups
        )
        session_id = str(model.get("device_session_id", ""))
        accepted = model.get("risk") == (session_id, revision)
        if not owners or not accepted:
            if not owners:
                model["risk"] = None
            accepted = False
        risk_possible = bool(owners) and any(
            not blockers[field.metadata.name] and field.live_write
            for field in tunables
        )
        if accepted and not risk_possible:
            model["risk"] = None
            accepted = False
        current = dict(model.get("current", {}))
        desired = dict(model.get("desired", {}))
        live_values = dict(model.get("live", {}))
        statuses = dict(model.get("status", {}))
        unapplied = dict(model.get("unapplied", {}))
        kept: dict[str, tuple[str, str]] = {}
        active = str(self._device_tune_active or "")
        fields: dict[str, object] = {}
        for tunable in tunables:
            name = tunable.metadata.name
            desired_value, desired_unit = desired.get(name, (current.get(name), tunable.metadata.unit or ""))
            unit_pending = name in model.get("unit_requests", {})
            protected = tuple(blockers[name])
            if protected:
                editable = False
                reason = "Protected by " + ", ".join(protected)
            elif owners and not tunable.live_write:
                editable = False
                reason = "Stop the active Logic; this field is not live-writable"
            elif owners and not accepted:
                editable = False
                reason = "Accept risk to edit this unclaimed live-safe field"
            else:
                editable = True
                reason = ""
            applying = active == f"{key}:{name}"
            queued = (key, name) in self._device_tune_pending
            to_apply = (
                name in model.get("unit_drafts", ())
                or (desired_value, desired_unit) != (current.get(name), tunable.metadata.unit or "")
            )
            # A queued value dropped unseen (refused as the drain reached it,
            # or cancelled by a claim change) is the newest fact about its
            # field while that value is still to apply: a status written for
            # another field does not take it down, nor does a reading unless
            # it reached the value.  Every Apply of the field takes it down
            # first, and so does Desired typed back to Current, which leaves
            # nothing to apply; a field no longer declared keeps none.
            note = unapplied.get(name) if unit_pending or to_apply else None
            if note is not None:
                kept[name] = note
            status, severity = note if note is not None else statuses.get(
                name,
                (
                    ("Applying; latest queued", "task") if applying and queued else
                    ("Applying", "task") if applying else
                    ("Queued latest", "task") if queued else
                    ("Protected", "warning") if not editable else
                    ("Stands outside its bounds", "warning")
                    if _stands_outside(tunable.metadata, current.get(name)) else
                    ("Ready", "ready")
                ),
            )
            fields[name] = {
                "current": current.get(name),
                # The instrument's own fence, read once at Init and shown
                # read-only beside the bench window, so the operator can see
                # which of the two bounds a refused value ran into.
                "device_limits": tunable.device_limits,
                "desired": desired_value,
                "desired_unit": desired_unit,
                "editable": editable,
                "live_apply": bool(live_values.get(name, False)),
                # CAPABILITY and PERMISSION are two facts.  Collapsed into one
                # the view could not tell "not allowed just now" from "this
                # field has no live write at all", so a window limit that can
                # never be applied live still drew a switch to not press.
                "live_capable": bool(tunable.live_write),
                "live_enabled": editable and tunable.live_write and not unit_pending,
                "apply_enabled": editable and not applying and not unit_pending and to_apply,
                "status": status,
                "severity": severity,
                "reason": reason,
            }
        model["unapplied"] = kept
        return {
            "owners": owners,
            "reason": (
                "No active Logic uses this device"
                if not owners else
                "Risk acceptance applies only to unclaimed live-safe fields"
            ),
            "risk_enabled": risk_possible,
            "risk_accepted": accepted,
            "fields": fields,
            "owner_revision": revision,
        }

    def _project_device_control(self, key: str) -> None:
        model = self._device_control_models.get(str(key))
        if model is None:
            return
        projection = self._device_control_projection(str(key))
        model["control"].set_projection(model["spec"], projection)
        model["shown_owner_revision"] = projection["owner_revision"]

    def _refresh_device_control_policies(self) -> None:
        """Project changed claims; local edits and command results project directly."""

        if self.session is None:
            return
        for key in tuple(self._device_control_models):
            model = self._device_control_models[key]
            if not model["control"].is_visible():
                continue
            if model.get("shown_owner_revision") == self.session.device_use.owner_revision(key):
                continue
            projection = self._device_control_projection(key)
            cancelled = False
            for pending in tuple(self._device_tune_pending):
                if pending[0] != key:
                    continue
                field = projection["fields"].get(pending[1])
                if not isinstance(field, dict) or not field.get("editable"):
                    self._device_tune_pending.pop(pending, None)
                    note = ("Cancelled because field ownership changed", "warning")
                    model.setdefault("unapplied", {})[pending[1]] = note
                    # The strip last said this value was queued.
                    model["control"].show_status(f"{pending[1]}: {note[0]}", note[1])
                    cancelled = True
            if cancelled:
                projection = self._device_control_projection(key)
            model["control"].set_projection(model["spec"], projection)
            model["shown_owner_revision"] = projection["owner_revision"]

    def _set_device_control_risk(self, key: str, accepted: bool) -> None:
        model = self._device_control_models.get(str(key))
        session = self.session
        if model is None or session is None:
            return
        if accepted:
            # Bound to what the operator was SHOWN: a claim that changed
            # since the last projection is not one they accepted.
            model["risk"] = (
                str(model.get("device_session_id", "")),
                int(model["shown_owner_revision"]),
            )
        else:
            model["risk"] = None
        self._project_device_control(str(key))

    def _set_device_control_desired(
        self, key: str, field: str, value: object, unit: str
    ) -> None:
        model = self._device_control_models.get(str(key))
        if model is None or str(field) not in dict(model.get("current", {})):
            return
        desired = dict(model.get("desired", {}))
        if desired[str(field)][1] != str(unit):
            model.setdefault("unit_drafts", set()).add(str(field))
        desired[str(field)] = (value, str(unit))
        model["desired"] = desired
        self._project_device_control(str(key))

    def _set_device_control_unit(self, key: str, field: str, unit: str) -> None:
        model = self._device_control_models.get(str(key))
        if model is None or str(field) not in model["desired"]:
            return
        model.setdefault("unit_requests", {})[str(field)] = str(unit)
        self._project_device_control(str(key))
        self._request_device_control_refresh(str(key))

    def _set_device_control_live(
        self, key: str, field: str, enabled: bool
    ) -> None:
        model = self._device_control_models.get(str(key))
        if model is None:
            return
        live = dict(model.get("live", {}))
        live[str(field)] = bool(enabled)
        model["live"] = live
        self._project_device_control(str(key))

    def _queue_device_tune(
        self, key: str, field: str, requested: object, unit: str
    ) -> tuple[str, str] | None:
        """Apply one value now, or queue it behind the tune on the worker.

        Returns the (text, severity) note a refused value was shown with;
        None when the value started, was queued, or has no control.
        """

        key, field = str(key), str(field)
        model = self._device_control_models.get(key)
        if model is None:
            return None
        if field in model.get("unit_requests", {}):
            note = ("Not applied while the display unit changed; Apply again", "warning")
            model["status"] = {field: note}
            self._project_device_control(key)
            return note
        model.setdefault("unapplied", {}).pop(field, None)
        projection = self._device_control_projection(key)
        selected = projection["fields"].get(field)
        if not isinstance(selected, dict) or not selected.get("editable"):
            note = (str(selected.get("reason", "Field is locked")) if isinstance(selected, dict) else "Unknown field", "warning")
            model["status"] = {field: note}
            self._project_device_control(key)
            return note
        if self._device_tune_active is not None:
            self._device_tune_pending[(key, field)] = (requested, unit)
            model["desired"][field] = (requested, unit)
            model["control"].show_status(
                f"queued latest {field}", "task"
            )
            self._project_device_control(key)
            return None
        return self._start_device_tune(key, field, requested, unit)

    def _start_device_tune(
        self, key: str, field: str, requested: object, unit: str
    ) -> tuple[str, str] | None:
        """Take the field's command and start the tune on the worker.

        Returns the (text, severity) note a refused value was shown with;
        None once the tune started, whatever it later finishes with.
        """

        from zlc_atom.authoring import AuthoringSchema, TunableField, read_tunable_in_unit, tune_in_unit
        from ..authoring_form import project_schema
        from ..device_use import DeviceClaim

        model = self._device_control_models[key]
        session = self.session
        run = self._device_worker_run
        if session is None or run is None:
            note = ("device tune worker is closed", "error")
            model["control"].show_status(*note)
            return note
        tunables = {item.metadata.name: item for item in model["tunables"]}
        try:
            tunable = tunables[field]
            projection = self._device_control_projection(key)
            selected = projection["fields"].get(field)
            if not isinstance(selected, dict) or not selected.get("editable"):
                reason = (
                    selected.get("reason", "Field is locked")
                    if isinstance(selected, dict)
                    else "Unknown field"
                )
                raise RuntimeError(str(reason))
            owners = tuple(projection["owners"])
            lease = session.device_use.acquire_field_command(
                model["control"],
                f"{key} control {field}",
                DeviceClaim(key, key, model["device"], (field,)),
                dependency_groups=tuple(
                    item.dependency_group for item in tunables.values()
                ),
                expected_owner_revision=int(projection["owner_revision"]),
                allow_while_logic=bool(owners and tunable.live_write),
            )
        except Exception as error:
            note = (str(error), "warning")
            model["status"] = {field: note}
            model["control"].show_status(*note)
            self._project_device_control(key)
            return note
        self._device_tune_active = f"{key}:{field}"
        model["status"] = {field: ("Applying", "task")}
        model["control"].show_status(f"applying {field}", "task")
        self._project_device_control(key)
        device = model["device"]
        with_logic = bool(owners)
        read_session = str(model.get("device_session_id", ""))
        display_units = {name: pair[1] for name, pair in model["desired"].items()}
        display_units[field] = unit

        def work() -> dict[str, object]:
            declared_before = tuple(device.tunable_fields())
            if any(not isinstance(item, TunableField) for item in declared_before):
                raise TypeError("device tunable_fields must contain TunableField values")
            current_tunable = {
                item.metadata.name: item for item in declared_before
            }.get(field)
            if current_tunable is None:
                raise ValueError(f"device no longer declares field {field!r}")
            if with_logic and not current_tunable.live_write:
                raise RuntimeError(
                    "field stopped being live-writable while Logic owns the device"
                )
            before = {
                item.metadata.name: item.current for item in declared_before
            }
            before_provenance = dict(device.settings_provenance())
            if str(before_provenance.get("device_session_id", "")).strip() != read_session:
                # A peer re-created the device: the reading this was applied
                # from -- its Current, its bounds, a risk accepted over it --
                # is another session's.
                raise RuntimeError("the device session changed; Refresh this control")
            effective = tune_in_unit(device, field, requested, unit)
            declared_after = tuple(device.tunable_fields())
            if any(not isinstance(item, TunableField) for item in declared_after):
                raise TypeError("device tunable_fields must contain TunableField values")
            after = {
                item.metadata.name: item.current for item in declared_after
            }
            after_provenance = dict(device.settings_provenance())
            before_session = str(
                before_provenance.get("device_session_id", "")
            ).strip()
            after_session = str(
                after_provenance.get("device_session_id", "")
            ).strip()
            before_epoch = before_provenance.get("settings_epoch")
            after_epoch = after_provenance.get("settings_epoch")
            if (
                not before_session
                or before_session != after_session
                or type(before_epoch) is not int
                or type(after_epoch) is not int
                or before_epoch < 0
            ):
                raise RuntimeError("device settings provenance changed identity")
            if after_epoch < before_epoch:
                raise RuntimeError("device settings epoch moved backwards")
            # Provenance stays in the device's canonical field units; only
            # the Control reading is projected into the requested spelling.
            displayed_fields = tuple(
                read_tunable_in_unit(device, item.metadata.name, display_units[item.metadata.name])
                if display_units.get(item.metadata.name) and display_units[item.metadata.name] != item.metadata.unit
                else item for item in declared_after
            )
            return {
                "previous": before[field],
                "new_effective": after[field],
                "before": before,
                "effective": effective,
                "current": after,
                "display_current": {item.metadata.name: item.current for item in displayed_fields},
                "tunables": displayed_fields,
                "before_provenance": before_provenance,
                "provenance": after_provenance,
            }

        def finish(result: dict[str, object] | None, error: BaseException | None) -> None:
            finish_error = error
            try:
                if result is not None:
                    session.record_device_tune(
                        device_key=key,
                        field=field,
                        requested=requested,
                        requested_unit=unit,
                        previous_effective=result["previous"],
                        new_effective=result["new_effective"],
                        before_provenance=result["before_provenance"],
                        after_provenance=result["provenance"],
                        previous_values=result["before"],
                        current_values=result["current"],
                        active_logic_owners=owners,
                    )
            except BaseException as provenance_error:
                finish_error = provenance_error
            finally:
                lease.release()
                self._device_tune_active = None
            if result is None or finish_error is not None:
                model["status"] = {field: (str(finish_error), "error")}
                model["control"].show_status(str(finish_error), "error")
            else:
                if any((item.metadata.unit or "") != model["desired"][item.metadata.name][1]
                       for item in result["tunables"]):
                    self._request_device_control_refresh(key)
                else:
                    model["current"] = dict(result["display_current"])
                    model["tunables"] = tuple(result["tunables"])
                    model["spec"] = project_schema(
                        AuthoringSchema(
                            tuple(item.metadata for item in model["tunables"])
                        )
                    )
                if ((key, field) not in self._device_tune_pending
                        and model["desired"][field] == (requested, unit)):
                    model["desired"][field] = (result["display_current"][field], unit)
                    model.setdefault("unit_drafts", set()).discard(field)
                model["device_session_id"] = str(
                    result["provenance"]["device_session_id"]
                )
                model["settings_epoch"] = int(
                    result["provenance"]["settings_epoch"]
                )
                model["status"] = {field: ("Applied", "ready")}
                model["control"].show_status(f"applied {field}", "idle")
            self._project_device_control(key)
            self._drain_device_tune_pending()

        try:
            run(
                work,
                lambda result: finish(dict(result), None),
                lambda error: finish(None, error),
            )
        except BaseException as error:
            finish(None, error)
        return None

    def _drain_device_tune_pending(self) -> None:
        # Until one starts: an entry refused here (its field locked, its
        # command refused, the worker closed or its display unit changed
        # meanwhile) is reported on its control, and must not park the rest
        # until some unrelated tune happens to finish.  The operator last saw
        # that value queued, so a note stays on its field while the value is
        # still to apply: the next entry -- often another field of the same
        # device -- writes its own status in this same turn, before anything
        # is painted.  The note says what happened to the value, with the
        # refusal's words as its reason, so it stays true once the lock or
        # lease that refused it is gone; the unit note is worded so already.
        # The control's strip, which last said the value was queued (or, for
        # a lease, said the refusal in the present tense), says the same note:
        # when the tune that ran belonged to another device's control, no
        # later line on this one would.
        while self._device_tune_active is None and self._device_tune_pending:
            (key, field), (requested, unit) = next(iter(self._device_tune_pending.items()))
            self._device_tune_pending.pop((key, field), None)
            model = self._device_control_models.get(key)
            if model is None:
                continue
            converting = field in model.get("unit_requests", {})
            refused = self._queue_device_tune(key, field, requested, unit)
            if refused is not None:
                text, severity = refused
                model.get("status", {}).pop(field, None)
                note = (
                    refused if converting else
                    (f"Queued value not applied ({text}); Apply again", severity)
                )
                model.setdefault("unapplied", {})[field] = note
                model["control"].show_status(f"{field}: {note[0]}", note[1])
                self._project_device_control(key)

    def _guard_control_gesture(self, control, what, action):
        """One generic-control gesture, unable to kill the bench.

        Same law as the console's view-signal guard; the flow outlives
        every control window, so a plain closure carries no lifetime risk,
        and the report lands on the control the operator was touching.
        Driving the underlying methods directly (as the tests do) still
        raises.
        """

        def guarded(*args):
            try:
                action(*args)
            except Exception as error:  # noqa: BLE001 -- the boundary IS total
                _LOG.exception("device control %s failed", what)
                try:
                    control.show_status(
                        f"internal error in {what}: "
                        f"{type(error).__name__}: {error}",
                        "error",
                    )
                except Exception:
                    _LOG.exception("device control status report failed")

        return guarded

    def _generic_control_close_guard(self, key: str, control: object) -> bool:
        if str(key) in self._device_refresh_active:
            control.show_status("device refresh is still running", "warning")
            return False
        active = str(self._device_tune_active or "")
        if not active.startswith(f"{str(key)}:"):
            for pending in tuple(self._device_tune_pending):
                if pending[0] == str(key):
                    self._device_tune_pending.pop(pending, None)
            return True
        control.show_status("device tune is still running", "warning")
        return False

    def _device_tune_idle(self) -> bool:
        """Whether nothing of the device worker's is still in flight.

        A control being READ for the first time is in flight as much as a
        refresh or a tune: its reading lands on the worker, and a session
        retired underneath it would be handed a control over devices that
        are gone.  There is no control window to say so on yet, so the
        Device Manager -- where the press was made -- says it.
        """

        if self._device_control_opening:
            key = next(iter(self._device_control_opening))
            self._report_device(
                f"{key}: a device control is still being read", "warning"
            )
            return False
        if self._device_refresh_active:
            key = next(iter(self._device_refresh_active))
            control = self.device_controls.get(key)
            if control is not None:
                control.show_status("device refresh is still running", "warning")
            return False
        active = self._device_tune_active
        if active is None:
            return True
        control = self.device_controls.get(str(active).partition(":")[0])
        if control is not None:
            control.show_status("device tune is still running", "warning")
        return False

    def _close_device_worker(self) -> bool:
        if not self._device_tune_idle():
            return False
        close = self._device_worker_close
        if close is None:
            return True
        if not close():
            return False
        self._device_worker_run = None
        self._device_worker_close = None
        return True

    def _retire_device_controls(
        self,
        device_keys: frozenset[str] | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Ask each control to close: the keys still closing, and those kept.

        A generic control closes at once.  A device editor (Pulse, SLM)
        refuses its first close and closes on its own time -- its work and
        plots must stop first -- and its ``closed`` release forgets it then.
        A control whose close says it is not under way was KEPT open (Cancel
        on an editor's discard question): nothing may wait for its ``closed``,
        and nothing after it is asked.  So the editors go first -- the
        discard question is theirs -- and a Cancel leaves every generic
        Control with its unapplied drafts, rather than closing them for a
        retire that then does not happen.
        """

        closing: list[str] = []
        kept: list[str] = []
        selected = tuple(
            (key, control)
            for key, control in tuple(self.device_controls.items())
            if device_keys is None or key in device_keys
        )
        for key, control in sorted(
            selected, key=lambda item: item[0] in self._device_control_models
        ):
            if kept:
                break
            under_way = control.close()
            if self.device_controls.get(key) is not control:
                continue
            if not control.is_visible():
                self._forget_device_control(key)
            else:
                (closing if under_way else kept).append(key)
        return tuple(closing), tuple(kept)

    def _device_controls_closed(self) -> bool:
        """Close every control before anything stops; whether all have.

        One still closing is waited for: its ``released`` resumes a pending
        shutdown, and a closing composition retries on the beat.  One kept
        open cancels the shutdown and the close outright.  Left pending,
        they fired on their own whenever that editor closed, hours later,
        and a closing composition asked its question again on every beat.
        """

        closing, kept = self._retire_device_controls()
        if kept:
            self._device_shutdown_pending = False
            self._closing_all = False
            self._report_device(
                f"shutdown cancelled: the {', '.join(kept)} control stayed open",
                "warning",
            )
            return False
        return not closing

    def _prepare_session_reconcile(
        self,
        session: object,
        config: object,
        close_keys: frozenset[str],
    ):
        """Stop only users of affected leaves, then return the worker half."""

        plan = session.plan_device_reconcile(
            config,
            close_keys=frozenset(close_keys),
        )
        from ..device_use import DeviceUseBusy

        affected = frozenset(plan.affected_keys)
        # Before anything closes: a command the barrier below refuses (a
        # remote publication, a tune, a PulseGUI run) refuses the change
        # here, not after its Controls were closed for it.
        holders = session.device_use.command_holders(tuple(sorted(affected)))
        if holders:
            raise DeviceUseBusy(holders)
        # So does a Control still reading its device, which holds no command:
        # its close refuses, and found only after the editors were asked, an
        # editor was closed for a change that then did not happen.
        reading = sorted(affected & (self._device_refresh_active | self._device_control_opening))
        if reading:
            raise RuntimeError(
                f"the {', '.join(reading)} control is still reading its device; "
                "press again when it is done"
            )
        # Before anything stops: a running Logic stopped for a change that
        # then waits on an editor's close was stopped for nothing.
        closing, kept = self._retire_device_controls(affected)
        if kept:
            raise RuntimeError(
                f"the {', '.join(kept)} control stayed open; nothing changed"
            )
        if closing:
            raise RuntimeError(
                f"closing the {', '.join(closing)} control first; "
                "press again once it has closed"
            )
        barrier = None
        if affected:
            barrier = session.device_use.begin_maintenance(
                self,
                "Device Manager change",
                tuple(sorted(affected)),
            )

        def work() -> object:
            try:
                if barrier is not None:
                    # Logic stop callbacks run on the GUI owner turn above;
                    # their leases release asynchronously at their normal
                    # cleanup boundary.  Never close a device before that.
                    barrier.wait(30.0)
                session.reconcile_devices(plan)
                return session
            finally:
                if barrier is not None:
                    barrier.release()

        return work

    def _session_reconciled(self, session: object) -> None:
        """Refresh device-dependent drafts without replacing TaskConsole."""

        installed = session.installation.devices
        stale_controls = frozenset(
            key
            for key, device in self._device_control_devices.items()
            if key not in installed or installed[key].device is not device
        )
        if stale_controls:
            self._retire_device_controls(stale_controls)
        labels = session.device_labels
        for key, control in self.device_controls.items():
            control.set_device_label(labels.get(key, key))
        if self.console_presenter is not None:
            self.console_presenter.installation_changed()

    def _console_owner_ready(self) -> None:
        if self._device_shutdown_pending and self.devices is not None:
            self.devices.presenter.shutdown_active()
            return
        if self.console is not None:
            self.console.close_later()

    def _continue_device_shutdown(self) -> None:
        if self._device_shutdown_pending and self.devices is not None:
            self.devices.presenter.shutdown_active()

    def _prepare_session_shutdown(self, session: object) -> bool:
        if not self._device_tune_idle():
            return False
        self._device_shutdown_pending = True
        # Controls first, while the console they draw in is still open; one
        # still closing resumes this shutdown when it has (``released``).
        if not self._device_controls_closed():
            return False
        presenter = self.console_presenter
        if presenter is not None and not presenter.close():
            return False
        if self.timer is not None:
            self.timer.stop()
        return True

    def _shutdown_session(self, session: object) -> None:
        session.close()

    def _session_shutdown_complete(self, session: object) -> None:
        self.session = None
        self.console_presenter = None
        self.timer = None
        self._device_shutdown_pending = False
        console = self.console
        if self._closing_all:
            if console is not None:
                console.close_later()
        else:
            self.console = None
            if console is not None:
                console.set_close_guard(lambda: True)
                console.close()
        if self.devices is not None and not self._closing_all:
            self.devices.restore()

    def _console_close_guard(self) -> bool:
        if self._closing_console:
            return False
        self._closing_console = True
        self._closing_all = True
        try:
            if (
                self.devices is not None
                and self.devices.presenter.device_operation_active
            ):
                if self.console_presenter is not None:
                    self.console_presenter._report(
                        "a Device Manager operation is still running",
                        severity="warning",
                    )
                return False
            if not self._device_tune_idle():
                return False
            # The controls first here too, while the console is still whole:
            # an editor kept open cancels this close before the console has
            # stopped anything.
            if not self._device_controls_closed():
                return False
            if self.console_presenter is not None and not self.console_presenter.close():
                return False
            if self.devices is not None:
                if not self.devices.presenter.shutdown_active():
                    return False
                if not self.devices.presenter.close():
                    return False
                if not self._close_device_worker():
                    return False
                self.devices.close()
            self.console = None
            return True
        except BaseException:
            self._closing_all = False
            return False
        finally:
            self._closing_console = False

    def _device_manager_close_guard(self) -> bool:
        """Retire the whole composition before its root window disappears."""

        self._closing_all = True
        if self.devices is not None and self.devices.presenter.device_operation_active:
            return False
        if not self._device_tune_idle():
            return False
        try:
            if self.devices is not None and not self.devices.presenter.close():
                return False
            return self._close_device_worker()
        except BaseException:
            self._closing_all = False
            return False

    def close(self) -> bool:
        """Advance composition shutdown without waiting on the Qt owner."""

        self._closing_all = True
        if not self._device_tune_idle():
            return False
        if self.console is not None:
            self.console.close()
            return self.console is None
        elif self.devices is not None:
            if not self.devices.presenter.shutdown_active():
                return False
            if not self.devices.presenter.close():
                return False
            if not self._close_device_worker():
                return False
            self.devices.close()
        elif not self._close_device_worker():
            return False
        self.devices = None
        return True


def create_experiment_flow(
    *,
    workspace=None,
    template=None,
    window_ratio=None,
) -> ExperimentGuiFlow:
    """Open Device Manager; Init creates one shared on-demand GUI session."""

    return ExperimentGuiFlow(
        workspace=workspace,
        template=template,
        window_ratio=window_ratio,
    ).open()


def create_window(
    *,
    workspace=None,
    template=None,
    window_ratio=None,
):
    """Open only TaskConsole for notebook and acceptance-capture callers.

    This entry owns the session it creates. The experiment launcher does not
    use it: ``main`` uses :func:`create_experiment_flow` so DeviceManager
    creates one session; its cards open controls on demand.
    """

    from ..board import attach_qt, attach_qt_worker

    _space, session = open_experiment(workspace, template)
    # The window is opened by build_console, through zlc_ui's one entry: this
    # layer composes and wires, and no longer knows what a window is made of.
    window, presenter = build_console(
        session,
        window_ratio=window_ratio,
    )
    # The presenter's beat, not the board's: the board is one step of it, and
    # the cadence is the display clock's one wall-time base.
    timer = attach_qt(
        presenter.beat, interval_ms=_beat_interval_ms(presenter), board=presenter.board
    )
    run_session_close, close_session_worker = attach_qt_worker(
        "zlc-console-session-close"
    )
    session_closing = False
    session_closed = False

    def _guard() -> bool:
        """Let go BEFORE the window goes, and keep it if letting go failed.

        This window owns a render worker and whatever logic nodes are running.
        Releasing them on ``closed`` -- after the close is committed -- means
        a failure part-way leaves an open device set with no window left to
        reach it.  A close guard is
        the mechanism for exactly that: the X now
        does nothing until the owners are confirmed down, and a failure leaves
        the window up so the operator can try again.
        """

        nonlocal session_closing, session_closed
        if not presenter.close():
            return False
        timer.stop()
        if session_closed:
            return close_session_worker()
        if session_closing:
            return False
        # A failed device close remains visible and retryable from the same X.
        session_closing = True

        def closed(_result: object) -> None:
            nonlocal session_closing, session_closed
            session_closing = False
            session_closed = True
            window.close_later()

        def failed(error: BaseException) -> None:
            nonlocal session_closing
            session_closing = False
            window.show_status(f"session did not close: {error}", "error")

        try:
            run_session_close(session.close, closed, failed)
        except BaseException as error:
            failed(error)
            return False
        return False

    window.presenter = presenter
    window.session = session
    window.set_close_guard(_guard)
    return window


def _open_saved_figure(
    parent: object,
    start: str,
    *,
    workspace: object,
    monitor_render: object,
    editor_render: object,
) -> object | None:
    """Open one saved figure in its own window, over the console's workspace.

    The console does not become a viewer; it asks for one.  What a saved figure
    is, and how to read it, belongs to the viewer -- which needs no session and
    happily opens a file from another bench or another year.
    """

    from .figure_viewer import create_window as create_viewer_window

    path = parent.ask_open_path(
        "Open saved figure", start, "Saved figures (*.npz);;All files (*)"
    )
    if not path:
        return None
    # The viewer's own public entry, not a second assembly of it: one window
    # definition means the console cannot open a viewer that differs from the
    # one the viewer's launcher opens.
    return create_viewer_window(
        path=path,
        workspace=workspace,
        monitor_render=monitor_render,
        editor_render=editor_render,
    ).presenter


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    from zlc_ui import ensure_qt_app

    application = ensure_qt_app([])

    if arguments.check:
        # The same assembly, without a window: a smoke test, not acceptance.
        try:
            space, session = open_experiment(arguments.workspace, arguments.template)
        except (FileNotFoundError, NotADirectoryError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"workspace: {space.root}", flush=True)
        for key, failure in session.failures.items():
            print(f"warning: device {key!r} did not open: {failure}", file=sys.stderr)
        # The release covers building the console too.  A failure while the
        # window is being assembled is the whole point of a smoke check, and it
        # is exactly when a session left open holds a camera nobody can reach.
        presenter = None
        try:
            _view, presenter = build_console(
                session,
            )
            for _beat in range(3):
                presenter.beat()
            print(
                f"console ready: {len(presenter.panels)} panel(s), "
                f"{len(presenter.offered_signals())} more signal(s) offerable"
            )
            return 0
        finally:
            try:
                if presenter is not None:
                    # The GUI normally advances this non-blocking close
                    # through its owner turns.  --check has no event loop
                    # after return, so it finishes the same orderly close
                    # here instead of leaving the (daemonic) render children
                    # to be terminated at interpreter exit.
                    import time

                    deadline = time.monotonic() + 30.0
                    while not presenter.close() and time.monotonic() < deadline:
                        presenter.beat()
                        time.sleep(0.005)
                    if not presenter.close():
                        raise RuntimeError(
                            "TaskConsole render processes did not close"
                        )
            finally:
                # Tried even when the console did not close: only a node
                # still holding a device keeps the devices open.
                session.close()

    try:
        flow = create_experiment_flow(
            workspace=arguments.workspace,
            template=arguments.template,
        )
    except (FileNotFoundError, NotADirectoryError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"workspace: {flow.space.root}")
    try:
        return int(application.exec_())
    finally:
        flow.close()
