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
from pathlib import Path

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
        shutdown_session: Callable[[object], None] | None = None,
        on_device_open: Callable[[str], None] | None = None,
    ) -> None:
        if initial_config is not None and not isinstance(initial_config, InstallationConfig):
            raise TypeError("initial_config must be InstallationConfig or None")
        for value, name in (
            (initialize_session, "initialize_session"),
            (on_initialized, "on_initialized"),
            (shutdown_session, "shutdown_session"),
            (on_device_open, "on_device_open"),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
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
        self.discovered: tuple[DeviceInstanceConfig, ...] = ()
        self._baseline_devices: tuple[DeviceInstanceConfig, ...] = ()
        self.saved = True
        self.busy = False
        self._closed = False
        self._initial_config = initial_config
        self._initialize_session = initialize_session
        self._on_initialized = on_initialized
        self._shutdown_session = shutdown_session
        self._on_device_open = on_device_open
        self._active_session: object | None = None
        self._active_config: InstallationConfig | None = None
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
            self.devices = list(
                self._initial_config.devices
                if _use_initial and self._initial_config is not None
                else ()
            )
            self._baseline_devices = tuple(self.devices)
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
        except Exception as error:
            self._report(f"cannot read {self.path.name}: {error}", severity="error")
            return False
        self.devices = list(config.devices)
        self._baseline_devices = tuple(self.devices)
        self.saved = True
        self._show()
        self._report(f"{len(self.devices)} device(s) from {self.path.name}")
        return True

    @property
    def active_session(self) -> object | None:
        """The exact session produced by Init, or ``None`` before/after it."""

        return self._active_session

    def new_from_template(self, name: str) -> bool:
        """Replace the local draft from a domain-owned installation template."""

        try:
            config = installation_config_from_template(self.catalog, str(name))
        except (KeyError, TypeError, ValueError) as error:
            self._report(str(error), severity="warning")
            return False
        self.devices = list(config.devices)
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

        if tuple(self.devices) == self._baseline_devices:
            return False
        self.devices = list(self._baseline_devices)
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

        if self.busy:
            return False
        self.busy = True
        self._show()
        self._report("scanning hardware")
        found: list[DeviceInstanceConfig] = []
        failures: list[str] = []
        for descriptor in self.types.values():
            if descriptor.discover is None:
                continue
            try:
                found.extend(descriptor.discover())
            except Exception as error:
                failures.append(f"{descriptor.type_id}: {error}")
        self.discovered = tuple(found)
        self.busy = False
        self._show()
        message = f"discovered {len(found)} device(s)"
        if failures:
            message += "; " + "; ".join(failures)
        self._report(message, severity="warning" if failures else "task")
        return True

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
        """Rename a device.  The role IS its name, so both move together.

        A role is how everything else reaches it, so two devices cannot share
        one -- the second would silently shadow the first for every consumer.
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
            instance_id, lambda item: replace(item, instance_id=role, role=role)
        )

    def set_type(self, instance_id: str, type_id: str) -> bool:
        """Change what a device IS, keeping every setting its new type shares.

        A qCMOS and a Basler both have an exposure; changing between them and
        being handed a blank exposure box is how a bench ends up running at the
        default.  What the new type does not have is dropped, because it has
        nowhere to go.
        """

        descriptor = self.types.get(str(type_id))
        if descriptor is None:
            return False
        current = self._device(instance_id)
        if current is None or current.type_id == descriptor.type_id:
            return False
        kept = {
            name: value
            for name, value in current.parameters.items()
            if name in descriptor.authoring_schema.field_names
        }
        return self._replace(
            instance_id,
            lambda item: replace(
                item,
                type_id=descriptor.type_id,
                parameters=descriptor.authoring_schema.project_values(kept),
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

    # ------------------------------------------------------------ session life

    def toggle_lifecycle(self) -> bool:
        """Init the shared session, or shut down the exact active session."""

        if self._active_session is None:
            return self._initialize_active()
        return self.shutdown_active()

    def _initialize_active(self) -> bool:
        if self._closed:
            self._report("device manager is closed", severity="error")
            return False
        if self.busy:
            return False
        if self._initialize_session is None:
            self._report(
                "this Device Manager has no session initializer",
                severity="warning",
            )
            return False
        try:
            candidate = InstallationConfig(tuple(self.devices))
        except Exception as error:
            self._report(f"this apparatus cannot be initialized: {error}", severity="error")
            return False

        self.busy = True
        self._show()
        self._report("initializing devices")
        try:
            session = self._initialize_session(candidate)
            if session is None:
                raise RuntimeError("session initializer returned None")
        except Exception as error:
            self.busy = False
            self._show()
            self._report(f"devices did not initialize: {error}", severity="error")
            return False

        self._active_session = session
        self._active_config = candidate
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
        self.busy = True
        self._show()
        self._report("shutting down devices")
        try:
            self._retire_session(session)
        except Exception as error:
            self.busy = False
            self._show()
            self._report(f"devices did not shut down: {error}", severity="error")
            return False
        self._active_session = None
        self._active_config = None
        self.busy = False
        self._show()
        self._report("devices shut down")
        return True

    def _retire_session(self, session: object) -> None:
        if self._shutdown_session is not None:
            self._shutdown_session(session)
            return
        close = getattr(session, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------ saving

    def save(self) -> str:
        """Write the apparatus, refusing anything a session could not open.

        The refusal comes from InstallationConfig, which is the thing that
        decides what a legal apparatus is -- duplicate roles, missing fields.
        Checking it here instead would be a second opinion, and the one that
        matters is the one the session will use tomorrow.
        """

        try:
            config = InstallationConfig(tuple(self.devices))
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
        self.saved = tuple(self.devices) == self._baseline_devices
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
            # The schema says which fields exist; the file says which were
            # chosen.  A device stores only what was set, so a type that GAINS
            # a field leaves every apparatus written before it short of one --
            # and the form, rightly, refuses a partial set of keys.  Opening
            # the file then failed outright: an editor that cannot show a
            # saved apparatus is an editor that cannot fix it either.
            stored = dict(item.parameters)
            self.view.set_form_spec(
                item.instance_id,
                spec,
                tuple(
                    (
                        field.key,
                        display_value(stored.get(field.key, field.default)),
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
        restart = active and self._active_differs()
        self.view.set_lifecycle(
            (
                "Shutdown for restart"
                if restart
                else "Shutdown devices"
                if active
                else "Init devices"
            ),
            enabled=active or (self._initialize_session is not None and bool(self.devices)),
            active=active,
            busy=self.busy,
        )

    def _template_name(
        self,
        devices: Sequence[DeviceInstanceConfig],
    ) -> str | None:
        shape = tuple((item.instance_id, item.type_id) for item in devices)
        for name in installation_template_names():
            try:
                config = installation_config_from_template(self.catalog, name)
            except KeyError:
                continue
            if shape == tuple(
                (item.instance_id, item.type_id) for item in config.devices
            ):
                return name
        return None

    def _active_differs(self) -> bool:
        if self._active_config is None:
            return False
        try:
            candidate = InstallationConfig(tuple(self.devices))
        except Exception:
            return True
        return candidate != self._active_config

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
        if not self.shutdown_active():
            return False
        self._closed = True
        return True
