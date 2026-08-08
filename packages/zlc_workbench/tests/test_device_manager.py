"""Authoring an apparatus, with zlc_atom still deciding what a device is.

Headless: the presenter never imports Qt, so what is under test is the wiring
between a mute view and the apparatus file a session will open tomorrow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.install.configuration import (
    DeviceInstanceConfig,
    InstallationConfig,
    load_installation_config,
)
from zlc_workbench.authoring_form import project_schema, project_values
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
        self.load_requested = _Signal()
        self.save_as_requested = _Signal()
        self.cancel_requested = _Signal()
        self.lifecycle_requested = _Signal()
        self.device_remove_requested = _Signal()
        self.role_committed = _Signal()
        self.type_picked = _Signal()
        self.parameter_committed = _Signal()
        self.choices: tuple = ()
        self.devices: tuple = ()
        self.forms: dict = {}
        self.values: dict = {}
        self.status: list[tuple[str, str]] = []

    def set_apparatus(self, name: str, dirty: bool, saved: bool) -> None:
        self.apparatus = (str(name), bool(dirty), bool(saved))

    def set_device_choices(self, choices, unavailable=()) -> None:
        self.choices = tuple(choices)
        #: What this machine cannot offer, and why -- read back by the test
        #: that proves a missing vendor runtime is named rather than hidden.
        self.unavailable = tuple(unavailable)

    def set_templates(self, templates) -> None:
        self.templates = tuple(templates)

    def set_installation_source(self, source, detail) -> None:
        self.installation_source = (source, str(detail))

    def set_lifecycle(
        self,
        text: str,
        *,
        enabled: bool,
        active: bool,
        busy: bool = False,
    ) -> None:
        self.lifecycle = (str(text), bool(enabled), bool(active), bool(busy))

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

    from zlc_atom.install.discovery import discover_device_types

    offered = {key for _label, key in manager.view.choices}
    assert offered == {item.type_id for item in discover_device_types()}


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
    """A role is how everything else reaches a device.

    The second would silently shadow the first for every consumer, which is a
    bench that looks configured and is not.
    """

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.device_add_requested.emit("camera.dcam")
    assert [item.role for item in manager.devices] == ["camera", "camera2"]

    manager.view.role_committed.emit("camera2", "camera")

    assert [item.role for item in manager.devices] == ["camera", "camera2"]
    assert "already called" in manager.view.status[-1][1]


def test_changing_a_device_type_keeps_the_settings_the_new_type_shares(manager) -> None:
    """A qCMOS and a Basler both have an exposure.

    Being handed a blank exposure box on the way between them is how a bench
    ends up running at the default.
    """

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.values["camera"]["exposure_seconds"] = 0.005
    manager.view.parameter_committed.emit("camera", "exposure_seconds")
    assert manager.devices[0].parameters["exposure_seconds"] == 0.005

    manager.view.type_picked.emit("camera", "camera.dcam")

    assert manager.devices[0].type_id == "camera.dcam"
    assert manager.devices[0].parameters["exposure_seconds"] == 0.005
    assert "device_index" in manager.devices[0].parameters


def test_a_value_a_device_would_refuse_is_refused_while_it_is_on_screen(manager) -> None:
    """Not at save time, when the operator has moved on to something else."""

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.values["camera"]["seed"] = -1

    manager.view.parameter_committed.emit("camera", "seed")

    assert manager.devices[0].parameters["seed"] == 0
    assert manager.view.status[-1][0] == "warning"


def test_a_frame_shape_is_edited_as_one_fact(manager) -> None:
    """A shape with a half-edited width is not a state worth reaching."""

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.values["camera"]["frame_shape_yx"] = "64, 96"

    manager.view.parameter_committed.emit("camera", "frame_shape_yx")

    assert manager.devices[0].parameters["frame_shape_yx"] == [64, 96]


def test_what_is_saved_is_what_a_session_opens(manager, tmp_path) -> None:
    """The whole point.  An apparatus nothing can read is not an apparatus."""

    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.device_add_requested.emit("sequencer.virtual")

    manager.view.save_requested.emit()

    written = tmp_path / "apparatus.json"
    assert written.exists()
    reopened = load_installation_config(written)
    assert [item.type_id for item in reopened.devices] == [
        "camera.virtual",
        "sequencer.virtual",
    ]
    # Plain data, readable without any zlc_ package.
    assert json.loads(written.read_text(encoding="utf-8"))["format"] == "zlc.installation"


def test_reopening_shows_what_was_saved(manager, tmp_path) -> None:
    manager.view.device_add_requested.emit("camera.virtual")
    manager.view.values["camera"]["seed"] = 7
    manager.view.parameter_committed.emit("camera", "seed")
    manager.view.save_requested.emit()

    reopened = DeviceManagerPresenter(_ManagerView(), tmp_path / "apparatus.json")

    assert [item.role for item in reopened.devices] == ["camera"]
    assert reopened.devices[0].parameters["seed"] == 7
    assert reopened.view.values["camera"]["seed"] == 7


def test_an_apparatus_a_session_could_not_open_is_not_written(manager, tmp_path) -> None:
    """The refusal comes from the thing that decides what a legal apparatus is.

    Checking it here as well would be a second opinion, and the one that matters
    is the one the session will use tomorrow.
    """

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
        )
    )

    frozen = project_values(schema, {"n": "12", "x": "1.5", "s": "3 4"})

    assert frozen == {"n": 12, "x": 1.5, "s": [3, 4]}
    with pytest.raises(ValueError):
        project_values(schema, {"n": 1, "x": 1.0, "s": "3"})


def test_an_apparatus_saved_before_a_type_gained_a_field_still_opens(tmp_path) -> None:
    """The schema says which fields exist; the file says which were chosen.

    A device stores only what was set, so a type that GAINS a setting leaves
    every apparatus written before it short of one -- and the form, rightly,
    refuses a partial set of keys.  Opening the file then failed outright with
    "form values must have exact keys", which is the one situation an apparatus
    editor exists for: an editor that cannot show a saved apparatus cannot fix
    it either.  Found by opening the workspace the notebooks ship with.
    """

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
                        # As an older release wrote it: one setting, not four.
                        "parameters": {"seed": 7},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view = _ManagerView()
    presenter = DeviceManagerPresenter(view, path, confirm_overwrite=lambda _p: True)

    spec = view.forms["camera"]
    values = view.values["camera"]
    assert {field.key for field in spec.fields} == set(values), (
        "every declared field must arrive, filled from the schema when the "
        "file is silent about it"
    )
    assert values["seed"] == 7, "and what WAS stored still wins"
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

    v1 kept a dot and a [*] in the header and recomputed both on every edit.
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

    from zlc_atom.install.discovery import discover_device_types

    camera = next(
        item for item in discover_device_types() if item.type_id == "camera.virtual"
    )
    initial = InstallationConfig(
        (
            DeviceInstanceConfig(
                instance_id="camera",
                role="camera",
                type_id=camera.type_id,
                parameters=camera.authoring_schema.freeze({}),
            ),
        )
    )
    session = object()
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
    assert view.lifecycle[:3] == ("Shutdown for restart", True, True)
    manager.cancel()
    assert view.lifecycle[:3] == ("Shutdown devices", True, True)

    manager.toggle_lifecycle()
    assert manager.active_session is None
    assert shut_down == [session]
    assert view.lifecycle[:3] == ("Init devices", True, False)

    manager.shutdown_active()
    manager.close()
    assert shut_down == [session], "shutdown and close are idempotent"
