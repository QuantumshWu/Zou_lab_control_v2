"""Authoring an apparatus, with zlc_atom still deciding what a device is.

Headless: the presenter never imports Qt, so what is under test is the wiring
between a mute view and the apparatus file a session will open tomorrow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from zlc_ui import STATUS_SEVERITIES

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.install.configuration import (
    DeviceInstanceConfig,
    InstallationConfig,
    load_installation_config,
)
from zlc_workbench.authoring_form import project_schema
from zlc_workbench.device_manager import DeviceManagerPresenter


class _Signal:
    def __init__(self) -> None:
        self._listeners: list = []

    def connect(self, listener) -> None:
        self._listeners.append(listener)

    def emit(self, *args) -> None:
        for listener in list(self._listeners):
            listener(*args)


class _ManagerView:
    """The device manager's whole contract, with Qt taken out."""

    def __init__(self) -> None:
        self.device_add_requested = _Signal()
        self.save_requested = _Signal()
        self.template_selected = _Signal()
        self.discovery_requested = _Signal()
        self.discovered_add_requested = _Signal()
        self.load_requested = _Signal()
        self.save_as_requested = _Signal()
        self.cancel_requested = _Signal()
        self.lifecycle_requested = _Signal()
        self.device_remove_requested = _Signal()
        self.role_committed = _Signal()
        self.type_picked = _Signal()
        self.parameter_committed = _Signal()
        self.device_open_requested = _Signal()
        self.device_close_requested = _Signal()
        self.device_remote_toggled = _Signal()
        self.choices: tuple = ()
        self.devices: tuple = ()
        self.forms: dict = {}
        self.values: dict = {}
        self.status: list[tuple[str, str]] = []

    def set_discovery_enabled(self, enabled, reason="") -> None:
        self.discovery_enabled = (bool(enabled), str(reason))

    def set_discovered_devices(self, devices) -> None:
        self.discovered_devices = tuple(devices)

    def set_apparatus(self, name: str, dirty: bool, saved: bool) -> None:
        self.apparatus = (str(name), bool(dirty), bool(saved))

    def set_device_choices(self, choices, unavailable=()) -> None:
        self.choices = tuple(choices)
        #: What this machine cannot offer, and why -- read back by the test
        #: that proves a missing vendor runtime is named rather than hidden.
        self.unavailable = tuple(unavailable)

    def set_templates(self, templates) -> None:
        self.templates = tuple(templates)

    def set_remoted(self, instance_ids) -> None:
        self.remoted = tuple(instance_ids)

    def set_lifecycle(
        self,
        text: str,
        *,
        enabled: bool,
        active: bool,
        busy: bool = False,
        changed: bool = False,
    ) -> None:
        self.lifecycle = (
            str(text),
            bool(enabled),
            bool(active),
            bool(busy),
            bool(changed),
        )

    def set_loaded_devices(self, devices) -> None:
        self.loaded_devices = tuple(devices)

    def set_devices(self, devices) -> None:
        self.devices = tuple(devices)

    def set_form_spec(self, instance_id, spec, values) -> None:
        self.forms[str(instance_id)] = spec
        self.values[str(instance_id)] = dict(values)

    def read_values(self, instance_id):
        return tuple(self.values.get(str(instance_id), {}).items())

    def show_status(self, text: str, severity: str) -> None:
        # The double answers for the REAL strip, vocabulary included.  It
        # used to take any word at all, so a severity the shipped view
        # rejects sailed through every test here and only failed in the
        # operator's console -- where the rejection came out of a Qt slot
        # and killed the whole application.
        if str(severity) not in STATUS_SEVERITIES:
            raise ValueError(
                f"severity {severity!r} is not one of {STATUS_SEVERITIES}"
            )
        self.status.append((str(severity), str(text)))


@pytest.fixture
def manager(tmp_path):
    view = _ManagerView()
    return DeviceManagerPresenter(view, tmp_path / "apparatus.json")


def test_a_bench_with_no_apparatus_yet_is_answered_not_refused(manager) -> None:
    """Which is how every new bench starts, so it cannot be an error."""

    assert manager.devices == []
    assert "no apparatus" in manager.view.status[-1][1]
    assert manager.view.choices, "no device type could be added"


def test_the_types_offered_are_the_types_that_can_be_built(manager) -> None:
    """Not a list this window keeps.  A second catalog drifts from the real one."""

    from zlc_atom.install import discover_device_catalog

    offered = {key for _label, key, _domain in manager.view.choices}
    assert offered == {item.type_id for item in discover_device_catalog().available}


def test_one_catalog_snapshot_drives_choices_unavailable_and_templates(tmp_path) -> None:
    """The presenter projects one discovery result; it never discovers again."""

    from zlc_atom.install import (
        DeviceCatalogSnapshot,
        discover_device_catalog,
        installation_config_from_template,
    )

    catalog = discover_device_catalog()
    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        catalog=catalog,
    )

    assert set(manager.types) == {item.type_id for item in catalog.available}
    assert view.unavailable == tuple(
        (item.family, item.reason) for item in catalog.unavailable
    )
    assert isinstance(manager.catalog, DeviceCatalogSnapshot)
    assert manager.new_from_template("virtual") is True
    assert InstallationConfig(
        tuple(manager.devices), simulation=manager.simulation
    ) == installation_config_from_template(
        catalog, "virtual"
    )


def test_adding_a_device_sets_it_up_the_way_its_own_schema_says(manager) -> None:
    manager.view.device_add_requested.emit("camera.virtual")

    assert len(manager.devices) == 1
    added = manager.devices[0]
    assert added.role == "camera"
    # Every field its schema declares, at the schema's own defaults.
    assert set(added.parameters) == set(
        manager.types["camera.virtual"].authoring_schema.field_names
    )
    assert manager.view.forms["camera"].keys == tuple(added.parameters)


def test_two_devices_cannot_share_one_name(manager) -> None:
    """Roles are unique operator-facing names.

    Stable instance ids remain separate from these editable labels.
    """

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.device_add_requested.emit("camera.dcam")
    assert [item.role for item in manager.devices] == ["camera", "camera2"]

    manager.view.role_committed.emit("camera2", "camera")

    assert [item.role for item in manager.devices] == ["camera", "camera2"]
    assert "already called" in manager.view.status[-1][1]

    manager.view.role_committed.emit("camera2", "qCMOS")
    assert manager.devices[1].instance_id == "camera2"
    assert manager.devices[1].role == "qCMOS"


def test_changing_device_type_starts_from_that_types_own_defaults(manager) -> None:
    """Equal field names do not claim equal hardware meaning or units."""

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.values["camera"]["exposure_seconds"] = 0.005
    manager.view.parameter_committed.emit("camera", "exposure_seconds")
    assert manager.devices[0].parameters["exposure_seconds"] == 0.005

    manager.view.type_picked.emit("camera", "camera.dcam")

    assert manager.devices[0].type_id == "camera.dcam"
    assert dict(manager.devices[0].parameters) == manager.types[
        "camera.dcam"
    ].authoring_schema.project_values()


def test_a_value_a_device_would_refuse_is_refused_while_it_is_on_screen(manager) -> None:
    """Not at save time, when the operator has moved on to something else."""

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.values["camera"]["exposure_seconds"] = -1

    manager.view.parameter_committed.emit("camera", "exposure_seconds")

    assert manager.devices[0].parameters["exposure_seconds"] == 0.02
    assert manager.view.status[-1][0] == "warning"


def test_a_frame_shape_is_edited_as_one_fact(manager) -> None:
    """A shape with a half-edited width is not a state worth reaching."""

    manager.view.device_add_requested.emit("camera.virtual_mot")
    manager.view.values["camera"]["frame_shape_yx"] = "64, 96"

    manager.view.parameter_committed.emit("camera", "frame_shape_yx")

    assert manager.devices[0].parameters["frame_shape_yx"] == (64, 96)


def test_save_and_reopen_preserve_devices_and_root_simulation(manager, tmp_path) -> None:
    """Device forms and the hand-edited root mapping survive one save/reopen."""

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.device_add_requested.emit("sequencer.virtual")
    manager.simulation = {"seed": 7}

    manager.view.save_requested.emit()

    written = tmp_path / "apparatus.json"
    assert written.exists()
    reopened = load_installation_config(written)
    assert [item.type_id for item in reopened.devices] == [
        "camera.virtual",
        "sequencer.virtual",
    ]
    assert reopened.simulation["seed"] == 7
    reopened_manager = DeviceManagerPresenter(
        _ManagerView(),
        written,
    )
    assert reopened_manager.simulation["seed"] == 7
    assert "seed" not in reopened_manager.view.values["camera"]
    assert reopened_manager.close()
    # Plain data, readable without any zlc_ package.
    assert json.loads(written.read_text(encoding="utf-8"))["format"] == "zlc.installation"


def test_a_structurally_duplicate_apparatus_is_not_written(manager, tmp_path) -> None:
    """The exact file grammar refuses duplicate roles before writing."""

    manager.view.device_add_requested.emit("camera.virtual")
    # Reach past the presenter to forge the state a duplicate role would leave.
    manager.devices.append(manager.devices[0])

    manager.view.save_requested.emit()

    assert not (tmp_path / "apparatus.json").exists()
    assert manager.view.status[-1][0] == "error"


def test_a_declared_type_no_widget_can_edit_is_refused_rather_than_guessed() -> None:
    """Silently rendering it as text saves the wrong thing, and says nothing."""

    from zlc_atom.authoring import AuthoringField, AuthoringSchema

    schema = AuthoringSchema((AuthoringField("thing", "matrix", "Thing"),))

    with pytest.raises(ValueError, match="matrix"):
        project_schema(schema)


def test_an_edited_value_comes_back_in_the_type_its_device_declared() -> None:
    from zlc_atom.authoring import AuthoringField, AuthoringSchema

    schema = AuthoringSchema(
        (
            AuthoringField("n", "int", "N"),
            AuthoringField("x", "float", "X"),
            AuthoringField("s", "pair", "S"),
            AuthoringField(
                "factors", "numeric_tuple", "Factors", (0.5, 2.0)
            ),
        )
    )

    frozen = schema.project_values(
        {"n": "12", "x": "1.5", "s": "3 4", "factors": "0.25, 4"}
    )

    assert frozen == {
        "n": 12,
        "x": 1.5,
        "s": [3, 4],
        "factors": (0.25, 4.0),
    }
    form_field = project_schema(schema).fields[-1]
    assert form_field.kind == "text"
    assert form_field.default == "0.5, 2.0"
    assert form_field.description == "comma-separated finite numbers"
    with pytest.raises(ValueError):
        schema.project_values({"n": 1, "x": 1.0, "s": "3"})
    with pytest.raises(TypeError):
        schema.project_values({"n": 1.5, "x": 1.0, "s": "3 4"})
    with pytest.raises(TypeError):
        schema.project_values({"n": 1, "x": 1.0, "s": (3.5, 4)})
    with pytest.raises(TypeError, match="finite numbers"):
        schema.project_values(
            {"n": 1, "x": 1.0, "s": "3 4", "factors": (True, 2.0)}
        )
    with pytest.raises(ValueError, match="finite"):
        schema.project_values(
            {"n": 1, "x": 1.0, "s": "3 4", "factors": "nan, 2"}
        )


def test_choice_projection_keeps_owner_labels_and_typed_values() -> None:
    from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema

    schema = AuthoringSchema(
        (
            AuthoringField(
                "binning",
                "choice",
                "Binning",
                1,
                choices=(
                    AuthoringChoice(1, "1 × 1"),
                    AuthoringChoice(2, "2 × 2"),
                ),
            ),
        )
    )

    field = project_schema(schema).fields[0]
    assert tuple((choice.label, choice.value) for choice in field.choices) == (
        ("1 × 1", 1),
        ("2 × 2", 2),
    )
    assert schema.project_values({"binning": 2}) == {"binning": 2}


def test_installation_missing_required_fields_is_reported(tmp_path) -> None:

    path = tmp_path / "apparatus.json"
    path.write_text(
        json.dumps(
            {
                "format": "zlc.installation",
                "devices": [
                    {
                        "instance_id": "camera",
                        "role": "camera",
                        "type_id": "camera.virtual",
                        "parameters": {"seed": 7},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view = _ManagerView()
    presenter = DeviceManagerPresenter(view, path, confirm_overwrite=lambda _p: True)

    assert presenter.devices == []
    assert view.status[-1][0] == "error"
    assert "exactly" in view.status[-1][1]
    del presenter


def test_a_family_this_machine_cannot_build_is_named_with_its_reason(tmp_path, monkeypatch) -> None:
    """Absent and impossible look the same, and they are not the same.

    A lab machine is normally missing a vendor runtime -- the DCAM SDK is not
    installed on the bench this was written on.  A family that will not import
    used to be simply absent from the picker, which reads exactly like a family
    that does not exist, so the question "why is my camera type not here" had
    nowhere to be answered.
    """

    from zlc_atom.install import discovery

    real = discovery._modules()
    monkeypatch.setattr(
        discovery, "_modules", lambda: real + ("zlc_atom.devices.notinstalled.device_types",)
    )

    path = tmp_path / "apparatus.json"
    view = _ManagerView()
    DeviceManagerPresenter(view, path, confirm_overwrite=lambda _p: True)

    assert view.choices, "the families that DO import must still be offered"
    assert view.unavailable, "the one that does not must be named"
    family, reason = view.unavailable[0]
    assert family == "notinstalled"
    assert "ModuleNotFoundError" in reason


def test_the_window_says_whether_the_file_has_what_is_on_screen(tmp_path) -> None:
    """"Can I close this?" is a question the window has to answer.

    The compact header keeps a dot and a [*] and recomputes both on every edit.
    Here nothing said it at all: an apparatus edited and not saved looked
    exactly like one just opened, which is how an afternoon of wiring gets
    closed away.
    """

    path = tmp_path / "apparatus.json"
    view = _ManagerView()
    manager = DeviceManagerPresenter(view, path, confirm_overwrite=lambda _p: True)

    name, dirty, saved = view.apparatus
    assert name == "apparatus.json"
    assert (dirty, saved) == (False, False), "nothing edited, nothing on disk"

    manager.add_device("camera.virtual")
    _name, dirty, saved = view.apparatus
    assert dirty is True, "an edit the file does not have must show"

    manager.save()
    _name, dirty, saved = view.apparatus
    assert (dirty, saved) == (False, True), "and saving must clear it"


def test_a_standalone_editor_without_a_session_factory_cannot_fake_init(tmp_path) -> None:
    """No callback means edit-only, never a temporary build-and-release path."""

    path = tmp_path / "apparatus.json"
    view = _ManagerView()
    manager = DeviceManagerPresenter(view, path, confirm_overwrite=lambda _p: True)
    manager.add_device("camera.virtual")

    assert view.lifecycle[:3] == ("Init devices", False, False)
    assert manager.toggle_lifecycle() is False
    assert manager.active_session is None


def test_init_holds_the_exact_session_until_explicit_shutdown(tmp_path) -> None:
    """Init is the shared Experiment boundary, not a build-and-release test."""

    from types import SimpleNamespace

    from zlc_atom.install import discover_device_catalog

    camera = next(
        item
        for item in discover_device_catalog().available
        if item.type_id == "camera.virtual"
    )
    initial = InstallationConfig(
        (
            DeviceInstanceConfig(
                instance_id="camera",
                role="camera",
                type_id=camera.type_id,
                parameters=camera.authoring_schema.project_values({}),
            ),
        )
    )
    session = SimpleNamespace(
        installation=SimpleNamespace(devices={"camera": object()}, failures={})
    )
    initialized: list[object] = []
    shut_down: list[object] = []
    candidates: list[InstallationConfig] = []

    def initialize(candidate: InstallationConfig) -> object:
        candidates.append(candidate)
        return session

    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=initialize,
        on_initialized=initialized.append,
        shutdown_session=shut_down.append,
    )

    assert tuple(manager.devices) == initial.devices
    assert manager.active_session is None
    manager.toggle_lifecycle()

    assert candidates == [initial]
    assert manager.active_session is session
    assert initialized == [session]
    assert shut_down == [], "Init must retain the session for both experiment windows"
    assert view.lifecycle[:3] == ("Shutdown devices", True, True)

    manager.set_role("camera", "camera_edited")
    assert view.lifecycle[:3] == ("Apply device changes", True, True)
    assert view.lifecycle[4] is True
    manager.cancel()
    assert view.lifecycle[:3] == ("Shutdown devices", True, True)

    manager.toggle_lifecycle()
    assert manager.active_session is None
    assert shut_down == [session]
    assert view.lifecycle[:3] == ("Init devices", True, False)

    manager.shutdown_active()
    manager.close()
    assert shut_down == [session], "shutdown and close are idempotent"


def test_an_active_draft_change_reconciles_without_replacing_or_shutting_session(
    tmp_path,
) -> None:
    from zlc_atom.install import discover_device_catalog

    camera = next(
        item
        for item in discover_device_catalog().available
        if item.type_id == "camera.virtual"
    )
    initial = InstallationConfig(
        (
            DeviceInstanceConfig(
                instance_id="camera",
                role="camera",
                type_id=camera.type_id,
                parameters=camera.authoring_schema.project_values({}),
            ),
        )
    )
    session = SimpleNamespace(
        installation=SimpleNamespace(devices={"camera": object()}, failures={})
    )
    prepared: list[tuple[object, InstallationConfig, tuple[str, ...]]] = []
    reconciled: list[object] = []
    shut_down: list[object] = []

    def prepare(active, candidate, close_keys):
        prepared.append((active, candidate, close_keys))
        return lambda: active

    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=lambda _candidate: session,
        prepare_reconcile=prepare,
        on_reconciled=reconciled.append,
        shutdown_session=shut_down.append,
    )
    assert manager.toggle_lifecycle() is True

    view.values["camera"]["exposure_seconds"] = 0.03
    assert manager.commit_parameters("camera", "exposure_seconds") is True
    assert view.lifecycle[:3] == ("Apply device changes", True, True)
    assert manager.toggle_lifecycle() is True

    assert len(prepared) == 1
    active, candidate, close_keys = prepared[0]
    assert active is session
    assert candidate.devices[0].parameters["exposure_seconds"] == 0.03
    assert close_keys == ()
    assert manager.active_session is session
    assert reconciled == [session]
    assert shut_down == []
    assert view.loaded_devices == (
        ("camera", "camera", "camera.virtual"),
    )
    assert view.lifecycle[:3] == ("Shutdown devices", True, True)


def test_loaded_close_targets_one_key_and_missing_device_remains_applyable(
    tmp_path,
) -> None:
    from zlc_atom.install import discover_device_catalog

    descriptors = {
        item.type_id: item for item in discover_device_catalog().available
    }
    initial = InstallationConfig(
        tuple(
            DeviceInstanceConfig(
                instance_id=key,
                role=key,
                type_id=type_id,
                parameters=descriptors[type_id].authoring_schema.project_values({}),
            )
            for key, type_id in (
                ("camera", "camera.virtual"),
                ("sequencer", "sequencer.virtual"),
            )
        )
    )
    installed = {"camera": object(), "sequencer": object()}
    session = SimpleNamespace(
        installation=SimpleNamespace(devices=installed, failures={})
    )
    prepared: list[tuple[InstallationConfig, tuple[str, ...]]] = []
    shut_down: list[object] = []

    def prepare(active, candidate, close_keys):
        assert active is session
        prepared.append((candidate, close_keys))

        def work():
            if close_keys:
                for key in close_keys:
                    installed.pop(key)
            else:
                for item in candidate.devices:
                    installed.setdefault(item.instance_id, object())
            return active

        return work

    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=lambda _candidate: session,
        prepare_reconcile=prepare,
        shutdown_session=shut_down.append,
    )
    assert manager.toggle_lifecycle() is True

    view.device_close_requested.emit("camera")

    assert prepared == [(initial, ("camera",))]
    assert set(installed) == {"sequencer"}
    assert view.loaded_devices == (
        ("sequencer", "sequencer", "sequencer.virtual"),
    )
    assert view.lifecycle[:3] == ("Apply device changes", True, True)
    assert manager.active_session is session
    assert shut_down == []

    assert manager.toggle_lifecycle() is True
    assert prepared[-1] == (initial, ())
    assert set(installed) == {"camera", "sequencer"}
    assert view.lifecycle[:3] == ("Shutdown devices", True, True)
    assert manager.active_session is session
    assert shut_down == []


def test_reconcile_failure_keeps_the_active_session_and_apply_state(tmp_path) -> None:
    from zlc_atom.install import discover_device_catalog

    camera = next(
        item
        for item in discover_device_catalog().available
        if item.type_id == "camera.virtual"
    )
    initial = InstallationConfig(
        (
            DeviceInstanceConfig(
                instance_id="camera",
                role="camera",
                type_id=camera.type_id,
                parameters=camera.authoring_schema.project_values({}),
            ),
        )
    )
    session = SimpleNamespace(
        installation=SimpleNamespace(devices={"camera": object()}, failures={})
    )
    reconciled: list[object] = []
    shut_down: list[object] = []

    def prepare(_active, _candidate, _close_keys):
        def fail():
            raise RuntimeError("camera refused reconfigure")

        return fail

    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=lambda _candidate: session,
        prepare_reconcile=prepare,
        on_reconciled=reconciled.append,
        shutdown_session=shut_down.append,
    )
    assert manager.toggle_lifecycle() is True
    assert manager.set_role("camera", "edited") is True

    assert manager.toggle_lifecycle() is True

    assert manager.active_session is session
    assert manager.busy is False
    assert reconciled == []
    assert shut_down == []
    assert view.lifecycle[:3] == ("Apply device changes", True, True)
    assert view.status[-1] == (
        "error",
        "device changes did not apply: camera refused reconfigure",
    )


def test_partial_failure_adopts_effective_config_and_refreshes_views(tmp_path) -> None:
    from zlc_atom.install import discover_device_catalog

    descriptors = {
        item.type_id: item for item in discover_device_catalog().available
    }
    initial = InstallationConfig(
        tuple(
            DeviceInstanceConfig(
                key,
                key,
                type_id,
                descriptors[type_id].authoring_schema.project_values({}),
            )
            for key, type_id in (
                ("camera", "camera.virtual"),
                ("sequencer", "sequencer.virtual"),
            )
        )
    )
    effective = InstallationConfig((initial.devices[1],), simulation=initial.simulation)
    installed = {"camera": object(), "sequencer": object()}
    session = SimpleNamespace(
        installation=SimpleNamespace(devices=installed, failures={}),
        installation_config=initial,
    )
    refreshed: list[object] = []

    def prepare(active, _candidate, _close_keys):
        def work():
            installed.pop("camera")
            active.installation_config = effective
            raise RuntimeError("camera close failed after another leaf closed")

        return work

    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=lambda _candidate: session,
        prepare_reconcile=prepare,
        on_reconciled=refreshed.append,
    )
    assert manager.toggle_lifecycle() is True
    assert manager.close_device("camera") is True

    assert refreshed == [session]
    assert manager._active_config == effective
    assert view.loaded_devices == (("sequencer", "sequencer", "sequencer.virtual"),)
    assert view.lifecycle[0] == "Apply device changes"
    assert "camera close failed" in view.status[-1][1]


def test_projection_refresh_failure_retries_without_running_hardware_again(tmp_path) -> None:
    from zlc_atom.install import discover_device_catalog

    camera = next(
        item
        for item in discover_device_catalog().available
        if item.type_id == "camera.virtual"
    )
    initial = InstallationConfig(
        (
            DeviceInstanceConfig(
                "camera",
                "camera",
                camera.type_id,
                camera.authoring_schema.project_values({}),
            ),
        )
    )
    session = SimpleNamespace(
        installation=SimpleNamespace(devices={"camera": object()}, failures={}),
        installation_config=initial,
    )
    hardware_calls: list[bool] = []
    refresh_calls: list[bool] = []

    def prepare(active, candidate, _close_keys):
        def work():
            hardware_calls.append(True)
            active.installation_config = candidate
            return active

        return work

    def refresh(_active):
        refresh_calls.append(True)
        if len(refresh_calls) == 1:
            raise RuntimeError("projection failed")

    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=lambda _candidate: session,
        prepare_reconcile=prepare,
        on_reconciled=refresh,
    )
    assert manager.toggle_lifecycle() is True
    assert manager.set_role("camera", "qCMOS") is True
    assert manager.toggle_lifecycle() is True
    assert hardware_calls == [True]
    assert view.lifecycle[0] == "Refresh device views"
    assert manager.close_device("camera") is False
    assert hardware_calls == [True]

    assert manager.toggle_lifecycle() is True
    assert hardware_calls == [True]
    assert refresh_calls == [True, True]
    assert view.lifecycle[0] == "Shutdown devices"


def test_a_loaded_card_forwards_control_to_the_session_window_owner(tmp_path) -> None:
    """DeviceManager emits identity; it neither embeds controls nor tunes hardware."""

    from types import SimpleNamespace

    from zlc_atom.install import discover_device_catalog

    descriptors = {
        item.type_id: item for item in discover_device_catalog().available
    }
    camera = descriptors["camera.virtual"]
    sequencer = descriptors["sequencer.virtual"]
    session = SimpleNamespace(
        installation=SimpleNamespace(
            devices={"camera": object()},
            failures={"sequencer": RuntimeError("not connected")},
        )
    )
    opened: list[str] = []
    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=InstallationConfig(
            (
                DeviceInstanceConfig(
                    instance_id="camera",
                    role="camera",
                    type_id="camera.virtual",
                    parameters=camera.authoring_schema.project_values({}),
                ),
                DeviceInstanceConfig(
                    instance_id="sequencer",
                    role="sequencer",
                    type_id="sequencer.virtual",
                    parameters=sequencer.authoring_schema.project_values({}),
                ),
            )
        ),
        initialize_session=lambda _candidate: session,
        on_device_open=opened.append,
    )

    assert manager.open_device("camera") is False
    assert opened == []
    assert manager.toggle_lifecycle() is True
    assert manager.active_session is session
    assert view.loaded_devices == (("camera", "camera", "camera.virtual"),)
    assert "1 device(s) initialized: camera · 1 failed" in view.status[-1][1]

    view.device_open_requested.emit("camera")
    assert opened == ["camera"]

    view.device_open_requested.emit("sequencer")
    assert opened == ["camera"], "configured but failed leaves are not loaded"
    assert "no loaded device" in view.status[-1][1]

    view.device_open_requested.emit("missing")
    assert opened == ["camera"]
    assert "no loaded device" in view.status[-1][1]


def test_scan_families_share_one_total_deadline(tmp_path, monkeypatch) -> None:
    import zlc_workbench.device_manager as tested_module

    deadline = 0.03
    monkeypatch.setattr(tested_module, "_FAMILY_SCAN_DEADLINE_SECONDS", deadline)
    manager = DeviceManagerPresenter(_ManagerView(), tmp_path / "apparatus.json")

    def slow(delay: float):
        def discover():
            time.sleep(delay)
            return ()

        return discover

    manager.types = {
        f"slow-{index}": SimpleNamespace(
            type_id=f"slow-{index}",
            discover=slow(delay),
        )
        for index, delay in enumerate((0.06, 0.09, 0.12))
    }
    started = time.monotonic()
    _found, failures = manager._scan_families()
    elapsed = time.monotonic() - started

    assert elapsed < deadline * 2.2, elapsed
    assert len(failures) == 3
    assert manager.close() is False
    time.sleep(0.13)
    assert manager.close() is True


def test_busy_presenter_refuses_close_then_closes_its_worker_and_rejects_work(
    tmp_path,
) -> None:
    submitted: list[tuple] = []
    worker_closed: list[bool] = []

    def run(work, deliver, failed) -> None:
        submitted.append((work, deliver, failed))

    def close_worker() -> bool:
        worker_closed.append(True)
        return True

    manager = DeviceManagerPresenter(
        _ManagerView(),
        tmp_path / "apparatus.json",
        run_off_thread=run,
        close_worker=close_worker,
    )
    assert manager.discover() is True
    assert manager.busy is True and len(submitted) == 1
    assert manager.close() is False
    assert worker_closed == []

    work, deliver, failed = submitted.pop()
    try:
        result = work()
    except BaseException as error:
        failed(error)
    else:
        deliver(result)
    assert manager.busy is False
    assert manager.close() is True
    assert worker_closed == [True]
    assert manager.discover() is False


def test_remote_toggle_publishes_and_withdraws_on_the_fabric(tmp_path) -> None:
    """One click beside Control publishes; a second withdraws; unload withdraws.

    A tunable device (the virtual RF -- the real driver over a memory
    library) is served by the fabric's generic data plane; the published
    set always names exactly what this machine can still serve.
    """

    from zlc_atom.devices.remote.fabric import list_remote_devices
    from zlc_atom.devices.rf.vaunix_lms import VaunixLmsConfig
    from zlc_atom.devices.simulation.rf import virtual_rf_source
    from zlc_atom.install import discover_device_catalog

    rf_type = next(
        item
        for item in discover_device_catalog().available
        if item.type_id == "rf.virtual"
    )
    initial = InstallationConfig(
        (
            DeviceInstanceConfig(
                instance_id="rf",
                role="detuning",
                type_id=rf_type.type_id,
                parameters=rf_type.authoring_schema.project_values({}),
            ),
        )
    )
    source = virtual_rf_source(VaunixLmsConfig(serial=1001))
    session = SimpleNamespace(
        installation=SimpleNamespace(
            devices={"rf": SimpleNamespace(device=source)}, failures={}
        )
    )
    view = _ManagerView()
    manager = DeviceManagerPresenter(
        view,
        tmp_path / "apparatus.json",
        initial_config=initial,
        initialize_session=lambda _candidate: session,
    )
    assert manager.toggle_lifecycle() is True
    try:
        assert manager.toggle_remote("rf") is True
        assert view.remoted == ("rf",)
        announcer = manager._announcer
        assert announcer is not None
        records = list_remote_devices("127.0.0.1", announcer.port)
        assert [record["instance_id"] for record in records] == ["rf"]
        assert records[0]["tunable"] is True

        assert manager.toggle_remote("rf") is True
        assert view.remoted == ()
        assert list_remote_devices("127.0.0.1", announcer.port) == ()

        # Published again, then unloaded: the fabric follows the session.
        assert manager.toggle_remote("rf") is True
        session.installation.devices.clear()
        manager._show()
        assert view.remoted == ()
        assert list_remote_devices("127.0.0.1", announcer.port) == ()
    finally:
        if manager._announcer is not None:
            manager._announcer.close()
        manager.close()
