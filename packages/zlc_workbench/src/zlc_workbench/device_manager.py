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

Nothing here opens a device.  Writing down that the bench has a Basler camera at
index 2 is a different act from reaching for it, and an apparatus has to be
editable from a laptop with no hardware attached at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from zlc_atom.install.configuration import (
    DeviceInstanceConfig,
    InstallationConfig,
    load_installation_config,
    save_installation_config,
)
from zlc_atom.install.discovery import discover_device_types, unavailable_device_types
from .authoring_form import display_value, project_schema, project_values


__all__ = ["DeviceManagerPresenter"]


class DeviceManagerPresenter:
    """Wires a device-manager view to one apparatus file."""

    def __init__(
        self,
        view: object,
        path: str | Path,
        *,
        device_types: Sequence[Any] | None = None,
        confirm_overwrite: Callable[[str], bool] | None = None,
    ) -> None:
        self.view = view
        self.path = Path(path)
        # The real catalog, not a copy of it: a window that listed its own
        # device types would drift from the ones that can actually be built.
        #: Families this machine cannot build today, named so the picker can
        #: say why rather than leaving a hole where a type used to be.
        self._unavailable = () if device_types is not None else unavailable_device_types()
        self.types = {
            descriptor.type_id: descriptor
            for descriptor in (
                discover_device_types() if device_types is None else device_types
            )
        }
        self.devices: list[DeviceInstanceConfig] = []
        self.saved = True
        self._confirm_overwrite = confirm_overwrite
        self._connect()
        self.load()

    # ------------------------------------------------------------------ wiring

    def _connect(self) -> None:
        self.view.device_add_requested.connect(self.add_device)
        self.view.device_remove_requested.connect(self.remove_device)
        self.view.role_committed.connect(self.set_role)
        self.view.type_picked.connect(self.set_type)
        self.view.parameter_committed.connect(self.commit_parameters)
        self.view.save_requested.connect(self.save)
        self.view.test_requested.connect(self.test_devices)
        # What this machine cannot offer is named too, with the reason: a
        # family that will not import used to be simply absent, which reads
        # exactly like a family that does not exist.
        self.view.set_device_choices(
            tuple(
                (f"{descriptor.type_id}", descriptor.type_id)
                for descriptor in sorted(self.types.values(), key=lambda item: item.type_id)
            ),
            tuple(
                (value.family, value.reason)
                for value in sorted(self._unavailable, key=lambda item: item.module)
            ),
        )

    # ------------------------------------------------------------------- state

    def load(self) -> bool:
        """Read the apparatus, or start an empty one and say so.

        A missing file is the ordinary case -- it is how a new bench begins --
        so it is answered rather than raised.
        """

        if not self.path.exists():
            self.devices = []
            self.saved = True
            self._show()
            self._report(f"no apparatus at {self.path.name} yet; add devices and save")
            return False
        try:
            config = load_installation_config(self.path)
        except Exception as error:
            self._report(f"cannot read {self.path.name}: {error}", severity="error")
            return False
        self.devices = list(config.devices)
        self.saved = True
        self._show()
        self._report(f"{len(self.devices)} device(s) from {self.path.name}")
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
                parameters=descriptor.authoring_schema.freeze({}),
            )
        )
        self._touch(f"added {role} ({descriptor.type_id})")
        return role

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
                parameters=descriptor.authoring_schema.freeze(kept),
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
            frozen = project_values(schema, dict(self.view.read_values(str(instance_id))))
        except Exception as error:
            self._report(f"{current.role}: {error}", severity="warning")
            return False
        if frozen == dict(current.parameters):
            return False
        return self._replace(instance_id, lambda item: replace(item, parameters=frozen))

    # ------------------------------------------------------------------ saving

    def test_devices(self) -> bool:
        """Bring up what is on screen, say what answered, and let go.

        A written apparatus is a claim, and the claim is worth checking before
        a run rather than during one: a role that will not open fails here, in
        the window built for fixing it, instead of ten minutes into an
        experiment.  v1 kept the built devices and upgraded the window into a
        session; here it does not, because in this architecture a SESSION owns
        devices and a second owner is how a camera ends up held by nobody and
        everybody at once.  So this proves the configuration and releases it.
        """

        from zlc_atom.install import create_installation

        try:
            config = InstallationConfig(tuple(self.devices))
        except Exception as error:
            self._report(f"this apparatus cannot be built: {error}", severity="error")
            return False
        try:
            installation = create_installation(config.specs())
        except Exception as error:
            self._report(f"devices did not come up: {error}", severity="error")
            return False
        try:
            roles = tuple(sorted(item.role for item in self.devices))
            self._report(
                f"{len(roles)} device(s) came up: {', '.join(roles)}", severity="task"
            )
        finally:
            close = getattr(installation, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    self._report(f"released with a complaint: {error}", severity="warning")
        return True

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
        self.saved = True
        # The dot and the [*] follow the file, so the projection runs again.
        self._show()
        self._report(f"saved {len(self.devices)} device(s) to {written.name}", severity="task")
        return str(written)

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
        self.saved = False
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
                (item.instance_id, item.role, item.type_id) for item in self.devices
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

    def _report(self, text: str, *, severity: str = "task") -> None:
        """One line of what just happened.

        The severity vocabulary is the status strip's, and is not restated:
        idle, warning, task, error.
        """

        self.view.show_status(str(text), str(severity))

    def close(self) -> None:
        """Nothing is open.  An apparatus editor holds no devices, by design."""
