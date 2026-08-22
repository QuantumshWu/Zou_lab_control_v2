"""Editing the apparatus: which devices this bench has, and how each is set up.

An apparatus is the one thing a session cannot start without, and until now the
only way to write one was by hand.  ``apparatus.json`` is plain data on purpose
-- it survives a session, it can be diffed, it can be mailed to whoever is
debugging the bench tomorrow -- and this is the window that authors it.

The presenter decides nothing about devices.  What device types exist, what each
one needs told, and what a legal value is all belong to zlc_atom, which already
declares them: every device type carries an authoring schema, and that schema is
the form.  A window that listed its own device types would be a second catalog
to keep in step with the real one.

The draft remains plain data, but the v1 lifecycle boundary is restored: an
embedding application may inject the one operation that turns the current
``InstallationConfig`` into its shared Experiment/session.  On success this
presenter retains that exact object until explicit shutdown; it never performs
a temporary build-and-release test.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

from zlc_atom.install.configuration import (
    DeviceInstanceConfig,
    InstallationConfig,
    load_installation_config,
    save_installation_config,
)
from zlc_atom.install import (
    DeviceCatalogSnapshot,
    discover_device_catalog,
    installation_config_from_template,
    installation_template_names,
)
from .authoring_form import display_value, project_schema


__all__ = ["DeviceManagerPresenter"]


#: How long one hardware family may take to answer a scan.  Generous, because
#: a network dial legitimately takes seconds; bounded, because an unreachable
#: one must not hold up the families that ARE there.
_FAMILY_SCAN_DEADLINE_SECONDS = 20.0


def _run_inline(work, deliver, failed) -> None:
    """Run the work right here: the headless behaviour every test drives."""

    try:
        result = work()
    except BaseException as error:  # noqa: BLE001 -- reported, not swallowed
        failed(error)
    else:
        deliver(result)


class DeviceManagerPresenter:
    """Wires a device-manager view to one apparatus file."""

    def __init__(
        self,
        view: object,
        path: str | Path,
        *,
        catalog: DeviceCatalogSnapshot | None = None,
        confirm_overwrite: Callable[[str], bool] | None = None,
        initial_config: InstallationConfig | None = None,
        initialize_session: Callable[[InstallationConfig], object] | None = None,
        on_initialized: Callable[[object], None] | None = None,
        prepare_reconcile: Callable[
            [object, InstallationConfig, tuple[str, ...]], Callable[[], object]
        ] | None = None,
        on_reconciled: Callable[[object], None] | None = None,
        prepare_shutdown: Callable[[object], bool] | None = None,
        shutdown_session: Callable[[object], None] | None = None,
        on_shutdown: Callable[[object], None] | None = None,
        on_device_open: Callable[[str], None] | None = None,
        run_off_thread: Callable[..., None] | None = None,
        close_worker: Callable[[], bool] | None = None,
    ) -> None:
        if initial_config is not None and not isinstance(initial_config, InstallationConfig):
            raise TypeError("initial_config must be InstallationConfig or None")
        for value, name in (
            (initialize_session, "initialize_session"),
            (on_initialized, "on_initialized"),
            (prepare_reconcile, "prepare_reconcile"),
            (on_reconciled, "on_reconciled"),
            (prepare_shutdown, "prepare_shutdown"),
            (shutdown_session, "shutdown_session"),
            (on_shutdown, "on_shutdown"),
            (on_device_open, "on_device_open"),
            (run_off_thread, "run_off_thread"),
            (close_worker, "close_worker"),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        #: How the slow, Qt-free half of a button runs.  Synchronously here,
        #: which is what every headless test drives and what a notebook wants;
        #: a GUI composition passes one that runs the work on a worker thread
        #: and delivers the result back on the GUI thread.  Without it, both
        #: buttons ran vendor bring-up inside their click slot: the event loop
        #: never turned, so the busy state and the "scanning hardware" line
        #: below were never painted, and the window was a frozen rectangle for
        #: however long the hardware took.
        self._run_off_thread = _run_inline if run_off_thread is None else run_off_thread
        self._close_worker = (lambda: True) if close_worker is None else close_worker
        self.view = view
        self.path = Path(path)
        # The real catalog, not a copy of it: a window that listed its own
        # device types would drift from the ones that can actually be built.
        #: Families this machine cannot build today, named so the picker can
        #: say why rather than leaving a hole where a type used to be.
        self.catalog = catalog if catalog is not None else discover_device_catalog()
        if not isinstance(self.catalog, DeviceCatalogSnapshot):
            raise TypeError("catalog must be DeviceCatalogSnapshot or None")
        self._unavailable = self.catalog.unavailable
        self.types = {
            descriptor.type_id: descriptor
            for descriptor in self.catalog.available
        }
        self.devices: list[DeviceInstanceConfig] = []
        self.simulation = {}
        self.discovered: tuple[DeviceInstanceConfig, ...] = ()
        self._baseline_devices: tuple[DeviceInstanceConfig, ...] = ()
        self._baseline_simulation = {}
        self.saved = True
        self.busy = False
        self._closed = False
        self._scan_lock = Lock()
        self._scan_pool: ThreadPoolExecutor | None = None
        self._scan_pending: set[object] = set()
        self._initial_config = initial_config
        self._initialize_session = initialize_session
        self._on_initialized = on_initialized
        self._prepare_reconcile = prepare_reconcile
        self._on_reconciled = on_reconciled
        self._prepare_shutdown = prepare_shutdown
        self._shutdown_session = shutdown_session
        self._on_shutdown = on_shutdown
        self._on_device_open = on_device_open
        self._active_session: object | None = None
        self._active_config: InstallationConfig | None = None
        self._refresh_pending = False
        self._confirm_overwrite = confirm_overwrite
        self._connect()
        self.load(_use_initial=True)

    # ------------------------------------------------------------------ wiring

    def _connect(self) -> None:
        self.view.device_add_requested.connect(self.add_device)
        self.view.device_remove_requested.connect(self.remove_device)
        self.view.role_committed.connect(self.set_role)
        self.view.type_picked.connect(self.set_type)
        self.view.parameter_committed.connect(self.commit_parameters)
        self.view.template_selected.connect(self.new_from_template)
        self.view.discovery_requested.connect(self.discover)
        self.view.discovered_add_requested.connect(self.add_discovered)
        self.view.load_requested.connect(self.load_from_dialog)
        self.view.save_requested.connect(self.save)
        self.view.save_as_requested.connect(self.save_as)
        self.view.cancel_requested.connect(self.cancel)
        self.view.lifecycle_requested.connect(self.toggle_lifecycle)
        self.view.device_open_requested.connect(self.open_device)
        self.view.device_close_requested.connect(self.close_device)
        # What this machine cannot offer is named too, with the reason: a
        # family that will not import used to be simply absent, which reads
        # exactly like a family that does not exist.
        self.view.set_device_choices(
            tuple(
                (
                    f"{descriptor.type_id}",
                    descriptor.type_id,
                    descriptor.domain,
                )
                for descriptor in sorted(self.types.values(), key=lambda item: item.type_id)
            ),
            tuple(
                (value.family, value.reason)
                for value in sorted(self._unavailable, key=lambda item: item.module)
            ),
        )
        self.view.set_templates(
            tuple(
                (name.replace("_", " ").title(), name)
                for name in installation_template_names()
            )
        )
        self.view.set_discovery_enabled(
            any(descriptor.discover is not None for descriptor in self.types.values())
        )

    # ------------------------------------------------------------------- state

    def load(
        self,
        path: str | Path | None = None,
        *,
        _use_initial: bool = False,
    ) -> bool:
        """Read the apparatus, or start an empty one and say so.

        A missing file is the ordinary case -- it is how a new bench begins --
        so it is answered rather than raised.
        """

        if path is not None:
            self.path = Path(path)
        if not self.path.exists():
            initial = (
                self._initial_config
                if _use_initial and self._initial_config is not None
                else None
            )
            self.devices = list(
                () if initial is None else initial.devices
            )
            self.simulation = {} if initial is None else initial.simulation
            self._baseline_devices = tuple(self.devices)
            self._baseline_simulation = self.simulation
            self.saved = True
            self._show()
            if self.devices:
                self._report(
                    f"new {self._template_name(self.devices) or 'custom'} apparatus draft"
                )
            else:
                self._report(f"no apparatus at {self.path.name} yet; add devices and save")
            return False
        try:
            config = load_installation_config(self.path)
            devices = [self._canonical_device(item) for item in config.devices]
        except Exception as error:
            self._report(f"cannot read {self.path.name}: {error}", severity="error")
            return False
        self.devices = devices
        self.simulation = config.simulation
        self._baseline_devices = tuple(self.devices)
        self._baseline_simulation = self.simulation
        self.saved = True
        self._show()
        self._report(f"{len(self.devices)} device(s) from {self.path.name}")
        return True

    @property
    def active_session(self) -> object | None:
        """The exact session produced by Init, or ``None`` before/after it."""

        return self._active_session

    @property
    def device_operation_active(self) -> bool:
        """Whether controls must stay closed during change or refresh."""

        return bool(self.busy or self._refresh_pending)

    def new_from_template(self, name: str) -> bool:
        """Replace the local draft from a domain-owned installation template."""

        try:
            config = installation_config_from_template(self.catalog, str(name))
        except (KeyError, TypeError, ValueError) as error:
            self._report(str(error), severity="warning")
            return False
        self.devices = list(config.devices)
        self.simulation = config.simulation
        self._touch(f"new {name} apparatus draft")
        return True

    def load_from_dialog(self) -> bool:
        ask = getattr(self.view, "ask_open_path", None)
        if not callable(ask):
            self._report("this window cannot choose an apparatus file", severity="warning")
            return False
        chosen = ask(
            "Open apparatus",
            str(self.path.parent),
            "Apparatus (*.json);;All files (*)",
        )
        return bool(chosen) and self.load(chosen)

    def cancel(self) -> bool:
        """Discard local edits and restore the last loaded/saved baseline."""

        if (
            tuple(self.devices) == self._baseline_devices
            and self.simulation == self._baseline_simulation
        ):
            return False
        self.devices = list(self._baseline_devices)
        self.simulation = self._baseline_simulation
        self.saved = True
        self._show()
        self._report("discarded unsaved apparatus edits")
        return True

    def add_device(self, type_id: str) -> str:
        """Add one device of a type, set up the way its own schema says.

        Named after its role, which is what everything downstream asks for: a
        session wants "the camera", not instance four.
        """

        descriptor = self.types.get(str(type_id))
        if descriptor is None:
            self._report(f"no device type {type_id!r}", severity="warning")
            return ""
        role = self._free_name(descriptor.domain)
        self.devices.append(
            DeviceInstanceConfig(
                instance_id=role,
                role=role,
                type_id=descriptor.type_id,
                parameters=descriptor.authoring_schema.project_values({}),
            )
        )
        self._touch(f"added {role} ({descriptor.type_id})")
        return role

    def discover(self) -> bool:
        """Scan each hardware family without connecting or configuring it."""

        if self._closed:
            self._report("device manager is closed", severity="error")
            return False
        if self.busy or self._scan_active():
            return False
        self.busy = True
        self._show()
        self._report("scanning hardware")

        def deliver(result) -> None:
            found, failures = result
            self.discovered = tuple(found)
            self.busy = False
            self._show()
            message = f"discovered {len(found)} device(s)"
            if failures:
                message += "; " + "; ".join(failures)
            self._report(message, severity="warning" if failures else "task")

        def failed(error: BaseException) -> None:
            self.busy = False
            self._show()
            self._report(f"scan failed: {error}", severity="error")

        self._run_off_thread(self._scan_families, deliver, failed)
        return True

    def _scan_families(self):
        """Ask every family at once, and let none of them hold the others up.

        One unreachable thing used to cost its whole timeout before the next
        family was even asked, sequentially, with no way to give up: a dialled
        sequencer that answers nothing is five seconds, and a vendor
        enumeration that hangs is forever.  A family that has not answered by
        the deadline is reported as such and the scan goes on; its thread is
        left to finish on its own, because a blocked vendor call cannot be
        cancelled and pretending otherwise would misreport what stopped.

        Nothing here touches the view: this half is what runs off the GUI
        thread, and the deliver callback above is what runs back on it.
        """

        descriptors = [
            descriptor
            for descriptor in self.types.values()
            if descriptor.discover is not None
        ]
        found: list[DeviceInstanceConfig] = []
        failures: list[str] = []
        if not descriptors:
            return found, failures
        pool = ThreadPoolExecutor(
            max_workers=len(descriptors), thread_name_prefix="zlc-scan"
        )
        pending = {
            pool.submit(descriptor.discover): descriptor
            for descriptor in descriptors
        }
        with self._scan_lock:
            if self._scan_pending:
                pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError("a previous hardware scan is still running")
            self._scan_pool = pool
            self._scan_pending = set(pending)
        for future in pending:
            future.add_done_callback(self._scan_finished)
        done, unfinished = wait(
            tuple(pending),
            timeout=_FAMILY_SCAN_DEADLINE_SECONDS,
        )
        for future, descriptor in pending.items():
            if future in unfinished:
                failures.append(
                    f"{descriptor.type_id}: no answer within "
                    f"{_FAMILY_SCAN_DEADLINE_SECONDS:g}s"
                )
                continue
            try:
                found.extend(future.result())
            except Exception as error:
                failures.append(f"{descriptor.type_id}: {error}")
        return found, failures

    def _scan_finished(self, future: object) -> None:
        pool = None
        with self._scan_lock:
            self._scan_pending.discard(future)
            if not self._scan_pending:
                pool, self._scan_pool = self._scan_pool, None
        if pool is not None:
            pool.shutdown(wait=False)

    def _scan_active(self) -> bool:
        with self._scan_lock:
            return bool(self._scan_pending)


    def add_discovered(self, instance_id: str) -> str:
        candidate = next(
            (item for item in self.discovered if item.instance_id == str(instance_id)),
            None,
        )
        if candidate is None:
            return ""
        if any(
            item.instance_id == candidate.instance_id or item.role == candidate.role
            for item in self.devices
        ):
            self._report(f"{candidate.role} is already configured", severity="warning")
            return ""
        self.devices.append(candidate)
        self._touch(f"added discovered {candidate.role} ({candidate.type_id})")
        return candidate.instance_id

    def remove_device(self, instance_id: str) -> bool:
        before = len(self.devices)
        self.devices = [
            item for item in self.devices if item.instance_id != str(instance_id)
        ]
        if len(self.devices) == before:
            return False
        self._touch(f"removed {instance_id}")
        return True

    def set_role(self, instance_id: str, role: str) -> bool:
        """Change the operator-facing role without changing device identity.

        ``instance_id`` is the stable key used by Logic drafts, live leases and
        differential installation.  A display rename must not turn an
        unchanged physical device into remove-old/add-new.
        """

        role = str(role).strip()
        if not role:
            self._report("a device needs a name", severity="warning")
            self._show()
            return False
        if any(
            item.role == role and item.instance_id != str(instance_id)
            for item in self.devices
        ):
            self._report(f"another device is already called {role!r}", severity="warning")
            self._show()
            return False
        return self._replace(
            instance_id, lambda item: replace(item, role=role)
        )

    def set_type(self, instance_id: str, type_id: str) -> bool:
        """Change what a device is, starting from that type's own defaults.

        Equal field names do not establish equal hardware meaning, unit or
        safe range.  Carrying values by spelling silently transfers one
        device's command into another device's schema.
        """

        descriptor = self.types.get(str(type_id))
        if descriptor is None:
            return False
        current = self._device(instance_id)
        if current is None or current.type_id == descriptor.type_id:
            return False
        return self._replace(
            instance_id,
            lambda item: replace(
                item,
                type_id=descriptor.type_id,
                parameters=descriptor.authoring_schema.project_values(),
            ),
        )

    def commit_parameters(self, instance_id: str, _field: str = "") -> bool:
        """Take what is in the form, in the types the device declared.

        Whole-form rather than per-field: the schema validates a device's
        settings together -- an exposure that is legal alone can be illegal
        beside a readout speed -- so a half-applied form is not a state worth
        being able to reach.
        """

        current = self._device(instance_id)
        if current is None:
            return False
        schema = self.types[current.type_id].authoring_schema
        try:
            frozen = schema.project_values(
                dict(self.view.read_values(str(instance_id)))
            )
        except Exception as error:
            self._report(f"{current.role}: {error}", severity="warning")
            return False
        if frozen == dict(current.parameters):
            return False
        return self._replace(instance_id, lambda item: replace(item, parameters=frozen))

    # -------------------------------------------------------------- live bench

    def open_device(self, instance_id: str) -> bool:
        """Forward one loaded-card Control intent to the experiment owner."""

        key = str(instance_id)
        if self.busy or self._refresh_pending:
            self._report(
                "finish the current device change before opening a control",
                severity="warning",
            )
            return False
        session = self._active_session
        if session is None:
            self._report("initialize devices before opening a control", severity="warning")
            return False
        if key not in session.installation.devices:
            self._report(f"no loaded device {key!r}", severity="warning")
            return False
        if self._on_device_open is None:
            self._report("this Device Manager has no control-window owner", severity="warning")
            return False
        try:
            self._on_device_open(key)
        except Exception as error:
            self._report(f"{key}: {error}", severity="error")
            return False
        return True

    def close_device(self, instance_id: str) -> bool:
        """Request retirement of exactly one currently loaded device."""

        key = str(instance_id)
        if self.device_operation_active:
            self._report(
                "finish the current device change before closing another device",
                severity="warning",
            )
            return False
        session = self._active_session
        if session is None:
            self._report("initialize devices before closing one", severity="warning")
            return False
        if key not in session.installation.devices:
            self._report(f"no loaded device {key!r}", severity="warning")
            return False
        candidate = self._active_config
        if candidate is None:
            self._report("active session has no installation config", severity="error")
            return False
        return self._reconcile_active(candidate, close_keys=(key,))

    # ------------------------------------------------------------ session life

    def toggle_lifecycle(self) -> bool:
        """Init, apply a changed draft, or shut down an unchanged session."""

        if self._active_session is None:
            return self._initialize_active()
        if self._refresh_pending:
            return self._retry_reconcile_refresh()
        try:
            candidate = InstallationConfig(
                tuple(self.devices), simulation=self.simulation
            )
        except Exception as error:
            self._report(f"this apparatus cannot be applied: {error}", severity="error")
            return False
        if self._active_differs(candidate):
            return self._reconcile_active(candidate)
        return self.shutdown_active()

    def _reconcile_active(
        self,
        candidate: InstallationConfig,
        *,
        close_keys: tuple[str, ...] = (),
    ) -> bool:
        """Prepare on the owner, reconcile off-thread, then project atomically."""

        if self._closed:
            self._report("device manager is closed", severity="error")
            return False
        if self.device_operation_active or self._scan_active():
            return False
        session = self._active_session
        if session is None:
            return False
        if self._prepare_reconcile is None:
            self._report(
                "this Device Manager has no session reconciler",
                severity="warning",
            )
            return False
        keys = tuple(dict.fromkeys(str(key) for key in close_keys))
        if any(not key for key in keys):
            self._report("device close key must be non-empty", severity="error")
            return False
        effective_before = getattr(session, "installation_config", None)

        self.busy = True
        self._show()
        self._report(
            f"closing {', '.join(keys)}" if keys else "applying device changes"
        )

        def failed(error: BaseException) -> None:
            refresh_error = None
            effective = getattr(session, "installation_config", None)
            if (
                isinstance(effective, InstallationConfig)
                and effective is not effective_before
            ):
                self._active_config = effective
                try:
                    if self._on_reconciled is not None:
                        self._on_reconciled(session)
                except Exception as projection_error:
                    self._refresh_pending = True
                    refresh_error = projection_error
                else:
                    self._refresh_pending = False
            self.busy = False
            self._show()
            self._report(
                f"device changes did not apply: {error}"
                + (
                    f"; device views did not refresh: {refresh_error}"
                    if refresh_error is not None
                    else ""
                ),
                severity="error",
            )

        try:
            work = self._prepare_reconcile(session, candidate, keys)
            if not callable(work):
                raise TypeError("session reconciler preparation must return callable work")
        except BaseException as error:
            failed(error)
            return False

        def reconcile() -> object:
            reconciled = work()
            if reconciled is not session:
                raise RuntimeError("device reconcile replaced the active session")
            return reconciled

        def finished(reconciled: object) -> None:
            if self._active_session is not reconciled:
                failed(RuntimeError("device reconcile completed for another session"))
                return
            effective = getattr(reconciled, "installation_config", None)
            if not isinstance(effective, InstallationConfig):
                effective = candidate
            try:
                if self._on_reconciled is not None:
                    self._on_reconciled(reconciled)
            except Exception as error:
                self._active_config = effective
                self._refresh_pending = True
                self.busy = False
                self._show()
                self._report(
                    f"devices changed but windows did not refresh: {error}",
                    severity="error",
                )
                return
            self._active_config = effective
            self._refresh_pending = False
            self.busy = False
            self._show()
            loaded = len(reconciled.installation.devices)
            self._report(
                (
                    f"{', '.join(keys)} closed · {loaded} device(s) loaded"
                    if keys
                    else f"device changes applied · {loaded} device(s) loaded"
                )
            )

        try:
            self._run_off_thread(reconcile, finished, failed)
        except BaseException as error:
            failed(error)
            return False
        return True

    def _retry_reconcile_refresh(self) -> bool:
        """Retry only presentation after devices already changed successfully."""

        session = self._active_session
        if session is None or self.busy or not self._refresh_pending:
            return False
        try:
            if self._on_reconciled is not None:
                self._on_reconciled(session)
        except Exception as error:
            self._report(f"device views did not refresh: {error}", severity="error")
            return False
        self._refresh_pending = False
        self._show()
        self._report("device views refreshed")
        return True

    def _initialize_active(self) -> bool:
        if self._closed:
            self._report("device manager is closed", severity="error")
            return False
        if self.busy or self._scan_active():
            return False
        if self._initialize_session is None:
            self._report(
                "this Device Manager has no session initializer",
                severity="warning",
            )
            return False
        try:
            candidate = InstallationConfig(
                tuple(self.devices), simulation=self.simulation
            )
        except Exception as error:
            self._report(f"this apparatus cannot be initialized: {error}", severity="error")
            return False

        self.busy = True
        self._show()
        self._report("initializing devices")

        def build() -> object:
            # Opening devices is where the seconds are -- a dial that answers
            # nothing, a vendor runtime coming up -- and none of it touches a
            # window.  Run it off the GUI thread and the "initializing
            # devices" line above is actually painted while it happens.
            session = self._initialize_session(candidate)
            if session is None:
                raise RuntimeError("session initializer returned None")
            return session

        def failed(error: BaseException) -> None:
            self.busy = False
            self._show()
            self._report(f"devices did not initialize: {error}", severity="error")

        self._run_off_thread(
            build,
            lambda session: self._session_ready(session, candidate),
            failed,
        )
        return True

    def _session_ready(self, session: object, candidate: InstallationConfig) -> bool:
        """The half that must happen where the windows are."""

        self._active_session = session
        self._active_config = candidate
        self._refresh_pending = False
        try:
            if self._on_initialized is not None:
                self._on_initialized(session)
        except Exception as error:
            try:
                self._retire_session(session)
            except Exception as shutdown_error:
                self.busy = False
                self._show()
                self._report(
                    f"window startup failed: {error}; session shutdown failed: "
                    f"{shutdown_error}",
                    severity="error",
                )
                return False
            self._active_session = None
            self._active_config = None
            self._refresh_pending = False
            self.busy = False
            self._show()
            self._report(f"window startup failed: {error}", severity="error")
            return False

        self.busy = False
        self._show()
        loaded_keys = frozenset(session.installation.devices)
        roles = ", ".join(
            item.role for item in candidate.devices if item.instance_id in loaded_keys
        )
        failed_count = len(session.installation.failures)
        self._report(
            f"{len(loaded_keys)} device(s) initialized"
            + (f": {roles}" if roles else "")
            + (f" · {failed_count} failed" if failed_count else "")
        )
        return True

    def shutdown_active(self) -> bool:
        """Shut down the retained session once; keep it reachable on failure."""

        session = self._active_session
        if session is None:
            return True
        if self.busy:
            return False
        if self._prepare_shutdown is not None:
            try:
                prepared = self._prepare_shutdown(session)
            except Exception as error:
                self._report(f"devices did not prepare to shut down: {error}", severity="error")
                return False
            if not prepared:
                return False
        self.busy = True
        self._show()
        self._report("shutting down devices")

        completed = False

        def work() -> object:
            self._retire_session(session)
            return session

        def failed(error: BaseException) -> None:
            self.busy = False
            self._show()
            self._report(f"devices did not shut down: {error}", severity="error")

        def finished(retired: object) -> None:
            nonlocal completed
            if self._active_session is not retired:
                raise RuntimeError("another session replaced the one being shut down")
            self._active_session = None
            self._active_config = None
            self._refresh_pending = False
            self.busy = False
            if self._on_shutdown is not None:
                self._on_shutdown(retired)
            self._show()
            self._report("devices shut down")
            completed = True

        try:
            self._run_off_thread(work, finished, failed)
        except BaseException as error:
            failed(error)
        return completed

    def _retire_session(self, session: object) -> None:
        if self._shutdown_session is not None:
            self._shutdown_session(session)
            return
        close = getattr(session, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------ saving

    def save(self) -> str:
        """Write one structurally exact, plain-data apparatus document.

        InstallationConfig owns the file grammar: duplicate roles and values
        JSON cannot preserve are refused here.  Simulation semantics stay with
        the single world resolver and are checked by Init before device
        factories run; Workbench does not carry a second copy of that grammar.
        """

        try:
            config = InstallationConfig(
                tuple(self.devices), simulation=self.simulation
            )
        except Exception as error:
            self._report(f"this apparatus cannot be saved: {error}", severity="error")
            return ""
        if (
            self.path.exists()
            and self._confirm_overwrite is not None
            and not self._confirm_overwrite(str(self.path))
        ):
            self._report("not saved")
            return ""
        try:
            written = save_installation_config(config, self.path)
        except Exception as error:
            self._report(f"cannot write {self.path.name}: {error}", severity="error")
            return ""
        self._baseline_devices = tuple(self.devices)
        self._baseline_simulation = self.simulation
        self.saved = True
        # The dot and the [*] follow the file, so the projection runs again.
        self._show()
        self._report(f"saved {len(self.devices)} device(s) to {written.name}", severity="task")
        return str(written)

    def save_as(self) -> str:
        """Choose another plain apparatus JSON and make it the editing target."""

        ask = getattr(self.view, "ask_save_path", None)
        if not callable(ask):
            self._report("this window cannot choose where to save", severity="warning")
            return ""
        chosen = ask(
            "Save apparatus as",
            # The file being edited, name and all: "save as" starts from what
            # you have, and an empty name box makes every save begin by typing.
            str(self.path),
            "Apparatus (*.json);;All files (*)",
        )
        if not chosen:
            return ""
        target = Path(chosen)
        if target.suffix == "":
            target = target.with_suffix(".json")
        previous = self.path
        self.path = target
        written = self.save()
        if not written:
            self.path = previous
            self._show()
        return written

    # ----------------------------------------------------------------- private

    def _device(self, instance_id: str) -> DeviceInstanceConfig | None:
        return next(
            (item for item in self.devices if item.instance_id == str(instance_id)),
            None,
        )

    def _canonical_device(self, item: DeviceInstanceConfig) -> DeviceInstanceConfig:
        """Materialize one known type's defaults into the editable truth."""

        descriptor = self.types.get(item.type_id)
        if descriptor is None:
            return item
        return replace(
            item,
            parameters=descriptor.authoring_schema.project_values(item.parameters),
        )

    def _free_name(self, domain: str) -> str:
        taken = {item.role for item in self.devices}
        if domain not in taken:
            return domain
        index = 2
        while f"{domain}{index}" in taken:
            index += 1
        return f"{domain}{index}"

    def _replace(
        self,
        instance_id: str,
        edit: Callable[[DeviceInstanceConfig], DeviceInstanceConfig],
    ) -> bool:
        for index, item in enumerate(self.devices):
            if item.instance_id == str(instance_id):
                try:
                    self.devices[index] = edit(item)
                except Exception as error:
                    self._report(f"{item.role}: {error}", severity="warning")
                    self._show()
                    return False
                self._touch(f"{self.devices[index].role} updated")
                return True
        return False

    def _touch(self, message: str) -> None:
        self.saved = (
            tuple(self.devices) == self._baseline_devices
            and self.simulation == self._baseline_simulation
        )
        self._show()
        self._report(message)

    def _show(self) -> None:
        """Put the whole apparatus on screen, forms and all.

        Including whether the screen and the file agree.  Recomputed here, on
        every projection, rather than polled or remembered: a dirty flag that
        is set in one place and read in another is the flag that ends up
        saying "saved" over an edit nobody wrote down.
        """

        self.view.set_apparatus(
            self.path.name, dirty=not self.saved, saved=self.path.exists()
        )
        self.view.set_devices(
            tuple(
                (
                    item.instance_id,
                    item.role,
                    item.type_id,
                    (
                        self.types[item.type_id].domain
                        if item.type_id in self.types
                        else "Unavailable"
                    ),
                )
                for item in self.devices
            )
        )
        for item in self.devices:
            descriptor = self.types.get(item.type_id)
            if descriptor is None:
                continue
            spec = project_schema(descriptor.authoring_schema)
            self.view.set_form_spec(
                item.instance_id,
                spec,
                tuple(
                    (
                        field.key,
                        display_value(item.parameters[field.key]),
                    )
                    for field in spec.fields
                ),
            )
        template = self._template_name(self.devices)
        self.view.set_installation_source(
            template,
            (
                f"{template.replace('_', ' ').title()} template"
                if template is not None
                else "Custom apparatus"
            )
            + f" · {len(self.devices)} named device(s)",
        )
        loaded_keys = (
            frozenset()
            if self._active_session is None
            else frozenset(self._active_session.installation.devices)
        )
        active_devices = tuple(
            (item.instance_id, item.role, item.type_id)
            for item in (() if self._active_config is None else self._active_config.devices)
            if item.instance_id in loaded_keys
        )
        self.view.set_loaded_devices(active_devices)
        configured = {item.instance_id for item in self.devices}
        self.view.set_discovered_devices(
            tuple(
                (
                    item.instance_id,
                    item.role,
                    item.type_id,
                    item.instance_id in configured,
                )
                for item in self.discovered
            )
        )
        active = self._active_session is not None
        reconcile = active and self._active_differs()
        self.view.set_lifecycle(
            (
                "Refresh device views"
                if self._refresh_pending
                else "Apply device changes"
                if reconcile
                else "Shutdown devices"
                if active
                else "Init devices"
            ),
            enabled=active or (self._initialize_session is not None and bool(self.devices)),
            active=active,
            busy=self.busy,
            changed=bool(reconcile),
        )

    def _template_name(
        self,
        devices: Sequence[DeviceInstanceConfig],
    ) -> str | None:
        current = InstallationConfig(tuple(devices), simulation=self.simulation)
        for name in installation_template_names():
            try:
                config = installation_config_from_template(self.catalog, name)
            except KeyError:
                continue
            if current == config:
                return name
        return None

    def _active_differs(
        self,
        candidate: InstallationConfig | None = None,
    ) -> bool:
        if self._active_config is None:
            return False
        if self._refresh_pending:
            return True
        if candidate is None:
            try:
                candidate = InstallationConfig(
                    tuple(self.devices), simulation=self.simulation
                )
            except Exception:
                return True
        if candidate != self._active_config:
            return True
        session = self._active_session
        if session is None:
            return False
        loaded = frozenset(session.installation.devices)
        configured = frozenset(item.instance_id for item in candidate.devices)
        return loaded != configured

    def _report(self, text: str, *, severity: str = "task") -> None:
        """One line of what just happened.

        The severity vocabulary is the status strip's, and is not restated:
        idle, warning, task, error.
        """

        self.view.show_status(str(text), str(severity))

    def close(self) -> bool:
        """Idempotently retire the retained shared session and this presenter."""

        if self._closed:
            return True
        if self.busy:
            self._report("device operation is still running", severity="warning")
            return False
        if self._scan_active():
            self._report("vendor hardware scan is still running", severity="warning")
            return False
        if not self.shutdown_active():
            return False
        try:
            worker_closed = self._close_worker()
        except Exception as error:
            self._report(f"device worker did not close: {error}", severity="error")
            return False
        if not worker_closed:
            self._report("device worker is still running", severity="warning")
            return False
        self._closed = True
        return True
