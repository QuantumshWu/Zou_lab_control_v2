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
    """Return the one wall cadence owned by the display clock.

    ``HarmonicClock.advance`` credits one clock base per beat, so the wall-time
    truth of every panel's labeled refresh interval requires the timer to fire
    at exactly that base.  An independent override silently rescales every
    panel's admission cadence, so it is rejected rather than retained as a
    second clock truth.
    """

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


def build_panel_host(plot_input, state):
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

    import zlc_plot as plot
    from ..panel_catalog import task_console_fitting_spec
    from ..panel_state import project_panel_state

    snapshot = getattr(plot_input, "snapshot", plot_input)
    spec = task_console_fitting_spec(
        snapshot.block.schema, state.kind, state.cell_kind
    )
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
    return plot.build_figure_host(
        plot_input,
        spec,
        size=state.size,
        parameters=parameters,
    )


def build_console(session, *, window_ratio=None, request_close=None):
    """One console presenter over one session, with the view it drives."""

    from ..panel_sizes import install as install_panel_sizes

    install_panel_sizes()
    import zlc_plot as plot

    from zlc_ui import open_task_console

    from ..board import attach_qt_owner_turn, attach_qt_worker
    from ..console import ConsolePresenter
    from ..panel_catalog import task_console_fitting_spec

    def _panel_surface(host):
        """A board panel's widget STAGES its fronts; the board presents them.

        Auto-present would put each panel's pixels up the moment its own
        render lands, so two panels of one causal group (a camera frame and
        the occupancy derived from it) could show different shots.  With
        staging, the presenter presents each same-shot batch atomically and
        routes every non-batch render through the same present helpers.
        """

        return plot.Qt5PlotWidget(host, auto_present=False)

    # One call, one handle: this layer never names a widget class.
    view = open_task_console(
        title="TaskConsole@Zou lab",
        window_ratio=window_ratio,
        plot_surface=_panel_surface,
    )

    def _spec_for(snapshot, kind: str = "", cell_kind: str = ""):
        """The spec this data admits, as the chosen kind or as its own shape."""

        return task_console_fitting_spec(
            snapshot.block.schema, kind, cell_kind
        )

    run_save, close_save_worker = attach_qt_worker("zlc-panel-save")

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
        presenter = ConsolePresenter(
            session,
            view,
            make_host=build_panel_host,
            spec_for=_spec_for,
            open_saved=lambda start: _open_saved_figure(view, start),
            request_close=view.close_later if request_close is None else request_close,
            run_off_thread=run_save,
            close_worker=close_save_worker,
            review_points=_review_points,
            manual_axis=_manual_axis,
        )
    except BaseException:
        close_save_worker()
        raise
    # Completion-driven presentation: a finished render's wake hops to the GUI
    # thread and commits immediately, instead of waiting for the next beat.
    presenter.board.wake.set_notify(
        attach_qt_owner_turn(presenter.commit_surfaces)
    )
    return view, presenter


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
        self._device_control_models: dict[str, dict[str, object]] = {}
        self._device_control_risk: dict[str, tuple[str, int] | None] = {}
        self._device_refresh_active: set[str] = set()
        self._device_refresh_pending: set[str] = set()
        self._device_shutdown_pending = False

    def open(self) -> "ExperimentGuiFlow":
        from zlc_atom.install import (
            discover_device_catalog,
            installation_config_from_template,
        )

        from .device_manager import create_window as create_device_window
        from ..board import attach_qt_worker

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
        run, close = attach_qt_worker("zlc-devices")
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
            )
            timer = attach_qt(
                self._beat,
                interval_ms=_beat_interval_ms(presenter),
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

    def open_device_control(self, instance_id: str) -> object:
        """Open or raise the one control window for a loaded named device."""

        key = str(instance_id)
        if (
            self.devices is not None
            and self.devices.presenter.device_operation_active
        ):
            raise RuntimeError("a Device Manager operation is still running")
        existing = self.device_controls.get(key)
        if existing is not None:
            existing.restore()
            return existing
        session = self.session
        if session is None or self.catalog is None:
            raise RuntimeError("initialize devices before opening a control")
        try:
            leaf = session.installation.devices[key]
        except KeyError as error:
            raise KeyError(f"no loaded device {key!r}") from error
        descriptors = {item.type_id: item for item in self.catalog.available}
        descriptor = descriptors[leaf.type_id]
        if descriptor.control_factory is not None:
            control = descriptor.control_factory(
                session,
                key,
                window_ratio=self.window_ratio,
            )
        else:
            control = self._open_generic_control(key, leaf.device)
        self.device_controls[key] = control
        self._device_control_devices[key] = leaf.device

        def released() -> None:
            if self.device_controls.get(key) is control:
                self.device_controls.pop(key, None)
                self._device_control_devices.pop(key, None)
                self._device_control_models.pop(key, None)
                self._device_control_risk.pop(key, None)
                self._device_refresh_active.discard(key)
                self._device_refresh_pending.discard(key)
                for pending in tuple(self._device_tune_pending):
                    if pending[0] == key:
                        self._device_tune_pending.pop(pending, None)

        control.closed.connect(released)
        return control

    def _console_owner_window(self):
        """The console window a FRAME of this console belongs to.

        Only for a window that has no life of its own -- the generic
        settings frame below is one: it shows the settings of a card this
        console is already showing, so the desktop must not stack anything
        between them.  A device that brings its own EDITOR (the pulse
        editor, the SLM editor) opens an application window instead: it is
        launched from here and then lives on its own, so it is not raised,
        minimised or closed with this console, and claiming it would tie
        two peers together.
        """

        console = self.console
        return None if console is None else console.owner_window

    def _open_generic_control(self, key: str, device: object) -> object:
        from zlc_atom.authoring import AuthoringSchema
        from zlc_ui import open_device_control

        from ..authoring_form import project_schema

        session = self.session
        if session is None:
            raise RuntimeError("initialize devices before opening a control")
        empty_spec = project_schema(AuthoringSchema(()))
        control = open_device_control(
            owner=self._console_owner_window(),
            title=f"{key} control",
            spec=empty_spec,
            projection={
                "owners": (),
                "reason": "Reading device settings",
                "risk_enabled": False,
                "risk_accepted": False,
                "fields": {},
            },
        )
        self._device_control_models[key] = {
            "device": device,
            "control": control,
            "spec": empty_spec,
            "tunables": (),
            "current": {},
            "desired": {},
            "live": {},
            "status": {},
            "device_session_id": "",
        }
        self._device_control_risk[key] = None
        control.refresh_requested.connect(
            lambda selected=key: self._request_device_control_refresh(selected)
        )
        control.risk_toggled.connect(
            lambda accepted, selected=key: self._set_device_control_risk(
                selected, accepted
            )
        )
        control.field_desired_changed.connect(
            lambda field, value, selected=key: self._set_device_control_desired(
                selected, field, value
            )
        )
        control.field_live_apply_toggled.connect(
            lambda field, enabled, selected=key: self._set_device_control_live(
                selected, field, enabled
            )
        )
        control.field_apply_requested.connect(
            lambda field, value, selected=key: self._queue_device_tune(
                selected, field, value
            )
        )
        self._request_device_control_refresh(key)
        control.set_close_guard(
            lambda: self._generic_control_close_guard(key, control)
        )
        return control

    def _request_device_control_refresh(self, key: str) -> None:
        from zlc_atom.authoring import AuthoringSchema, TunableField
        from ..authoring_form import project_schema

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
        control.show_status("refreshing device settings", "task")

        def work() -> tuple[tuple[object, ...], dict[str, object], dict[str, object]]:
            declare = getattr(device, "tunable_fields", None)
            if not callable(declare):
                return (), {}, {}
            fields = tuple(declare())
            if any(not isinstance(field, TunableField) for field in fields):
                raise TypeError("device tunable_fields must contain TunableField values")
            tune = getattr(device, "tune", None)
            provenance_reader = getattr(device, "settings_provenance", None)
            if fields and (not callable(tune) or not callable(provenance_reader)):
                raise TypeError(
                    "a tunable device must provide tune and settings_provenance"
                )
            current = {
                field.metadata.name: field.current for field in fields
            }
            provenance = {} if not fields else dict(provenance_reader())
            names = tuple(field.metadata.name for field in fields)
            if set(current) != set(names):
                raise ValueError("tunable current values differ from field metadata")
            session_id = str(provenance.get("device_session_id", "")).strip()
            epoch = provenance.get("settings_epoch")
            if fields and (
                not session_id or type(epoch) is not int or epoch < 0
            ):
                raise ValueError("device settings provenance is invalid")
            return fields, current, provenance

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
                fields, current, provenance = result
                names = tuple(field.metadata.name for field in fields)
                session_id = (
                    "" if not fields else str(provenance["device_session_id"])
                )
                epoch = 0 if not fields else int(provenance["settings_epoch"])
                previous_session = str(model.get("device_session_id", ""))
                model["tunables"] = fields
                model["current"] = current
                model["spec"] = project_schema(
                    AuthoringSchema(tuple(field.metadata for field in fields))
                )
                if previous_session != session_id:
                    model["desired"] = dict(current)
                    model["live"] = {name: False for name in names}
                    self._device_control_risk[key] = None
                else:
                    desired = dict(model.get("desired", {}))
                    model["desired"] = {
                        name: desired.get(name, current[name]) for name in names
                    }
                    live = dict(model.get("live", {}))
                    model["live"] = {
                        name: bool(live.get(name, False)) for name in names
                    }
                model["device_session_id"] = session_id
                model["settings_epoch"] = epoch
                model["status"] = {}
                self._project_device_control(key)
                control.show_status(
                    "ready" if fields else "No runtime controls", "idle"
                )
            except BaseException as error:
                control.show_status(str(error), "error")
            finally:
                settled()

        def failed(error: BaseException) -> None:
            if self._device_control_models.get(key) is model:
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
        accepted = self._device_control_risk.get(key) == (session_id, revision)
        if not owners or not accepted:
            if not owners:
                self._device_control_risk[key] = None
            accepted = False
        risk_possible = bool(owners) and any(
            not blockers[field.metadata.name] and field.live_write
            for field in tunables
        )
        if accepted and not risk_possible:
            self._device_control_risk[key] = None
            accepted = False
        current = dict(model.get("current", {}))
        desired = dict(model.get("desired", {}))
        live_values = dict(model.get("live", {}))
        statuses = dict(model.get("status", {}))
        active = str(self._device_tune_active or "")
        fields: dict[str, object] = {}
        for tunable in tunables:
            name = tunable.metadata.name
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
            status, severity = statuses.get(
                name,
                (
                    ("Applying; latest queued", "task") if applying and queued else
                    ("Applying", "task") if applying else
                    ("Queued latest", "task") if queued else
                    ("Protected", "warning") if not editable else
                    ("Ready", "ready")
                ),
            )
            fields[name] = {
                "current": current.get(name),
                "desired": desired.get(name, current.get(name)),
                "editable": editable,
                "live_apply": bool(live_values.get(name, False)),
                "live_enabled": editable and tunable.live_write,
                "apply_enabled": (
                    editable
                    and not applying
                    and desired.get(name, current.get(name)) != current.get(name)
                ),
                "status": status,
                "severity": severity,
                "reason": reason,
            }
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
        model["control"].set_projection(
            model["spec"], self._device_control_projection(str(key))
        )

    def _refresh_device_control_policies(self) -> None:
        for key in tuple(self._device_control_models):
            projection = self._device_control_projection(key)
            for pending in tuple(self._device_tune_pending):
                if pending[0] != key:
                    continue
                field = projection["fields"].get(pending[1])
                if not isinstance(field, dict) or not field.get("editable"):
                    self._device_tune_pending.pop(pending, None)
                    self._device_control_models[key]["status"] = {
                        pending[1]: ("Cancelled because field ownership changed", "warning")
                    }
            self._project_device_control(key)

    def _set_device_control_risk(self, key: str, accepted: bool) -> None:
        model = self._device_control_models.get(str(key))
        session = self.session
        if model is None or session is None:
            return
        if accepted:
            projection = self._device_control_projection(str(key))
            self._device_control_risk[str(key)] = (
                str(model.get("device_session_id", "")),
                int(projection["owner_revision"]),
            )
        else:
            self._device_control_risk[str(key)] = None
        self._project_device_control(str(key))

    def _set_device_control_desired(
        self, key: str, field: str, value: object
    ) -> None:
        model = self._device_control_models.get(str(key))
        if model is None or str(field) not in dict(model.get("current", {})):
            return
        desired = dict(model.get("desired", {}))
        desired[str(field)] = value
        model["desired"] = desired
        self._project_device_control(str(key))

    def _set_device_control_live(
        self, key: str, field: str, enabled: bool
    ) -> None:
        model = self._device_control_models.get(str(key))
        if model is None:
            return
        live = dict(model.get("live", {}))
        live[str(field)] = bool(enabled)
        model["live"] = live

    def _queue_device_tune(self, key: str, field: str, requested: object) -> None:
        key, field = str(key), str(field)
        model = self._device_control_models.get(key)
        if model is None:
            return
        projection = self._device_control_projection(key)
        selected = projection["fields"].get(field)
        if not isinstance(selected, dict) or not selected.get("editable"):
            model["status"] = {field: (str(selected.get("reason", "Field is locked")) if isinstance(selected, dict) else "Unknown field", "warning")}
            self._project_device_control(key)
            return
        if self._device_tune_active is not None:
            self._device_tune_pending[(key, field)] = requested
            model["desired"][field] = requested
            model["control"].show_status(
                f"queued latest {field}", "task"
            )
            self._project_device_control(key)
            return
        self._start_device_tune(key, field, requested)

    def _start_device_tune(self, key: str, field: str, requested: object) -> None:
        from zlc_atom.authoring import AuthoringSchema, TunableField
        from ..authoring_form import project_schema
        from ..device_use import DeviceClaim

        model = self._device_control_models[key]
        session = self.session
        run = self._device_worker_run
        if session is None or run is None:
            model["control"].show_status("device tune worker is closed", "error")
            return
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
            model["status"] = {field: (str(error), "warning")}
            model["control"].show_status(str(error), "warning")
            self._project_device_control(key)
            return
        self._device_tune_active = f"{key}:{field}"
        model["status"] = {field: ("Applying", "task")}
        model["control"].show_status(f"applying {field}", "task")
        self._project_device_control(key)
        device = model["device"]
        with_logic = bool(owners)

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
            effective = device.tune(field, requested)
            declared_after = tuple(device.tunable_fields())
            if any(not isinstance(item, TunableField) for item in declared_after):
                raise TypeError("device tunable_fields must contain TunableField values")
            after = {
                item.metadata.name: item.current for item in declared_after
            }
            after_provenance = dict(device.settings_provenance())
            if after.get(field) != effective:
                raise RuntimeError("device tune return differs from authoritative readback")
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
            expected_epoch = before_epoch + int(before[field] != effective)
            if after_epoch != expected_epoch:
                raise RuntimeError(
                    "device settings epoch does not match the effective change"
                )
            return {
                "previous": before[field],
                "before": before,
                "effective": effective,
                "current": after,
                "tunables": declared_after,
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
                        previous_effective=result["previous"],
                        new_effective=result["effective"],
                        verified=True,
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
                model["current"] = dict(result["current"])
                model["tunables"] = tuple(result["tunables"])
                model["spec"] = project_schema(
                    AuthoringSchema(
                        tuple(item.metadata for item in model["tunables"])
                    )
                )
                if (key, field) not in self._device_tune_pending:
                    model["desired"][field] = result["effective"]
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

    def _drain_device_tune_pending(self) -> None:
        if self._device_tune_active is not None or not self._device_tune_pending:
            return
        (key, field), requested = next(iter(self._device_tune_pending.items()))
        self._device_tune_pending.pop((key, field), None)
        if key in self._device_control_models:
            self._queue_device_tune(key, field, requested)

    def _generic_control_close_guard(self, key: str, control: object) -> bool:
        if str(key) in self._device_refresh_active:
            control.show_status("device refresh is still running", "warning")
            return False
        active = str(self._device_tune_active or "")
        if not active.startswith(f"{str(key)}:"):
            return True
        control.show_status("device tune is still running", "warning")
        return False

    def _device_tune_idle(self) -> bool:
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
    ) -> None:
        for key, control in tuple(self.device_controls.items()):
            if device_keys is not None and key not in device_keys:
                continue
            control.close()
            if self.device_controls.get(key) is control:
                if control.is_visible():
                    raise RuntimeError(f"{key} control refused to close")
                self.device_controls.pop(key, None)
                self._device_control_devices.pop(key, None)
                self._device_control_models.pop(key, None)
                self._device_control_risk.pop(key, None)
                self._device_refresh_active.discard(key)
                self._device_refresh_pending.discard(key)
                for pending in tuple(self._device_tune_pending):
                    if pending[0] == key:
                        self._device_tune_pending.pop(pending, None)

    def _prepare_session_reconcile(
        self,
        session: object,
        config: object,
        close_keys: frozenset[str],
    ):
        """Stop only users of affected leaves, then return the worker half."""

        if self.session is not session:
            raise RuntimeError("DeviceManager tried to change another experiment session")
        plan = session.plan_device_reconcile(
            config,
            close_keys=frozenset(close_keys),
        )
        affected = frozenset(plan.affected_keys)
        barrier = None
        if affected:
            barrier = session.device_use.begin_maintenance(
                self,
                "Device Manager change",
                tuple(sorted(affected)),
            )
            try:
                self._retire_device_controls(affected)
            except BaseException:
                barrier.release()
                raise

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

        if self.session is not session:
            raise RuntimeError("another experiment session replaced the changed one")
        installed = session.installation.devices
        stale_controls = frozenset(
            key
            for key, device in self._device_control_devices.items()
            if key not in installed or installed[key].device is not device
        )
        if stale_controls:
            self._retire_device_controls(stale_controls)
        if self.console_presenter is not None:
            self.console_presenter.installation_changed()

    def _console_owner_ready(self) -> None:
        if self._device_shutdown_pending and self.devices is not None:
            self.devices.presenter.shutdown_active()
            return
        if self.console is not None:
            self.console.close_later()

    def _prepare_session_shutdown(self, session: object) -> bool:
        if self.session is not None and session is not self.session:
            raise RuntimeError("DeviceManager tried to retire another experiment session")
        if not self._device_tune_idle():
            return False
        self._device_shutdown_pending = True
        presenter = self.console_presenter
        if presenter is not None and not presenter.close():
            return False
        self._retire_device_controls()
        if self.timer is not None:
            self.timer.stop()
        return True

    def _shutdown_session(self, session: object) -> None:
        session.close()

    def _session_shutdown_complete(self, session: object) -> None:
        if self.session is not session:
            raise RuntimeError("another experiment session replaced the retired one")
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
            self._closing_all = True
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

        if self.devices is not None and self.devices.presenter.device_operation_active:
            return False
        if not self._device_tune_idle():
            return False
        self._closing_all = True
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
        presenter.beat, interval_ms=_beat_interval_ms(presenter)
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


def _open_saved_figure(parent: object, start: str) -> object | None:
    """Open one saved figure in its own window, over today's data folder.

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
    return create_viewer_window(path=path).presenter


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    from zlc_ui import ensure_qt_app

    application = ensure_qt_app([])

    if arguments.check:
        # The same assembly, without a window: a smoke test, not acceptance.
        try:
            space, session = open_experiment(arguments.workspace, arguments.template)
        except FileNotFoundError as error:
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
            if presenter is not None:
                presenter.close()
            session.close()

    try:
        flow = create_experiment_flow(
            workspace=arguments.workspace,
            template=arguments.template,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"workspace: {flow.space.root}")
    try:
        return int(application.exec_())
    finally:
        flow.close()
