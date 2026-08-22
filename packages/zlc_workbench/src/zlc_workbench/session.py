"""One experiment session, driven the same way from a notebook or a GUI.

This is the seam the whole split exists for.  A notebook calls these methods; a
presenter calls exactly the same ones in response to a button.  If the two ever
grow separate paths, the GUI and the notebook stop being the same experiment and
every guarantee proved on one stops applying to the other.

The session composes; it does not compute.  Devices come from zlc_atom, the
signal plane from zlc_runtime, the pulse from the workspace, the archive format
from each owning package.  Nothing here decides anything about physics, about
rendering, or about how signals flow -- those have homes, and this is not one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date as _date
import json
from numbers import Integral
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from zlc_durable import atomic_write_bytes, day_folder, readable_json_bytes
from zlc_durable.paths import resolve_under

from .device_use import DeviceClaim, DeviceUseCoordinator
from .pulse_state import PulseEditorState, read_pulse

if TYPE_CHECKING:
    from zlc_atom.install import DeviceCatalogSnapshot
    from zlc_atom.install.configuration import InstallationConfig


__all__ = [
    "DeviceReconcilePlan",
    "ExperimentSession",
    "PulseEditorState",
    "Workspace",
    "read_pulse",
]


@dataclass(frozen=True)
class DeviceReconcilePlan:
    """One immutable, generation-checked change to a live installation."""

    source_installation: object = field(compare=False, repr=False)
    source_revision: int
    session_revision: int
    target_config: object
    retained_keys: tuple[str, ...]
    affected_keys: tuple[str, ...]
    build_keys: tuple[str, ...]
    world_changed: bool


def _connect_pulse(host: str, port: int, **kwargs: Any) -> object:
    """Reach a pulse server.  Imported here, and nowhere in the domain package."""

    from zlc_pulse import connect

    return connect(host, port, **kwargs)


def _same_json_value(left: object, right: object) -> bool:
    """Compare JSON trees without conflating booleans and numbers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _same_device_setup(left: object, right: object) -> bool:
    """Compare the factory inputs for two named-device declarations."""

    return (
        left.type_id == right.type_id
        and _same_json_value(
            left.to_dict()["parameters"],
            right.to_dict()["parameters"],
        )
    )


def _resolved_simulation(space: "Workspace", config: object) -> dict[str, object]:
    """Resolve the one path-valued simulation field exactly as initial open does."""

    simulation = dict(config.simulation)
    profile = simulation.get("world_profile", "")
    if isinstance(profile, str) and profile.strip():
        simulation["world_profile"] = str(
            resolve_under(space.root, profile.strip())
        )
    return simulation


def _seed_default_pulse(template: Path, packaged: bytes) -> None:
    """Create or canonicalize only a packaged default pulse.

    A differently authored or unreadable file is operator content and remains
    untouched.  An equivalent old rendering is not a second pulse: write it
    back through the one readable JSON serializer and durable file owner.
    """

    packaged_tree = json.loads(packaged.decode("utf-8"))
    canonical = readable_json_bytes(packaged_tree)
    if not template.exists():
        atomic_write_bytes(template, canonical)
        return
    try:
        existing = template.read_bytes()
        existing_tree = json.loads(existing.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if _same_json_value(existing_tree, packaged_tree) and existing != canonical:
        atomic_write_bytes(template, canonical)


def seed_packaged_pulses(pulses: Path) -> None:
    """Every packaged default pulse, present and canonical, in one place.

    Called from every session entry, not only the checkout's own home: a
    fresh experiment folder gets working calibration and scan templates, and
    the seeder never touches operator content -- it creates what is absent
    and canonicalizes only files that are byte-for-byte equivalent trees.
    """

    from zlc_atom.nodes import (
        calibration_pulse_template_bytes,
        scan_pulse_template_bytes,
        temperature_pulse_template_bytes,
    )

    pulses.mkdir(parents=True, exist_ok=True)
    _seed_default_pulse(
        pulses / Workspace.IMAGING_TEMPLATE, calibration_pulse_template_bytes()
    )
    _seed_default_pulse(
        pulses / "mot_field_template.json", scan_pulse_template_bytes()
    )
    _seed_default_pulse(
        pulses / "temperature_template.json", temperature_pulse_template_bytes()
    )


@dataclass(frozen=True)
class Workspace:
    """Where an experiment's own files live: its pulses and its saved data.

    Pulse definitions are experiment content, not library code -- they change as
    the experiment does -- so they live beside the data rather than inside an
    installed package, and the session is told where to find them.
    """

    root: Path

    def __init__(self, root: str | os.PathLike[str]) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"workspace root does not exist: {resolved}")
        object.__setattr__(self, "root", resolved)

    @property
    def pulses(self) -> Path:
        return self.root / "pulses"

    def pulse(self, name: str) -> Path:
        """Resolve one pulse stem or plain JSON filename inside ``pulses/``."""

        text = str(name).strip()
        requested = Path(text)
        if (
            not text
            or requested.is_absolute()
            or requested.parent != Path(".")
            or text != requested.name
            or requested.name in (".", "..")
        ):
            raise ValueError(f"pulse name must be a plain filename: {name!r}")
        if requested.suffix and requested.suffix.lower() != ".json":
            raise ValueError(f"pulse files must use a .json suffix: {name!r}")
        filename = requested.name if requested.suffix else f"{requested.name}.json"
        return self.pulses / filename

    @property
    def apparatus(self) -> Path:
        """The file describing what is in the lab today."""

        return self.root / "apparatus.json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    #: What makes a directory an experiment's, rather than just a directory.
    MARKERS = ("pulses", "apparatus.json")

    def prepare(self) -> "Workspace":
        """Create the directories a session writes into."""

        self.data.mkdir(parents=True, exist_ok=True)
        return self

    #: Where an experiment lives when nobody has said otherwise.  Overridable,
    #: because a lab that keeps its data elsewhere should say so once rather
    #: than pass a path to every launcher.
    HOME_VARIABLE = "ZLC_WORKSPACE"
    #: Inside the checkout, beside the code that drives it, and NOT tracked --
    #: .gitignore keeps it out, so a pull cannot replace a pulse and a reclone
    #: does not carry one machine's experiment to another.
    DEFAULT_HOME = "workspace"
    #: Calibration's authored project template.  It is merely seeded into the
    #: default workspace; opening devices and opening the ordinary Pulse Editor
    #: never load or compile it implicitly.
    IMAGING_TEMPLATE = "imaging_template.json"

    @classmethod
    def default(cls) -> "Workspace":
        """The one place an experiment lives when nothing else is nearer.

        In the checkout: ``<repo>/workspace/``, holding pulses/, data/ and
        apparatus.json.  One folder, wherever the launcher was double-clicked
        from, and it travels with the code that opens it.

        NOT the working directory, which is what this used to fall back on: a
        double-clicked launcher starts in the folder holding it, so "the
        default pulse folder" was a different folder depending on how the
        window was opened, and a saved pulse landed somewhere nobody would look
        for it again.

        It is created on demand.  A home that must be made by hand before the
        first save is a home that will not be there.
        """

        root = os.environ.get(cls.HOME_VARIABLE, "").strip()
        home = (
            Path(root).expanduser()
            if root
            else Path(__file__).resolve().parents[4] / cls.DEFAULT_HOME
        )
        seed_packaged_pulses(home / "pulses")
        return cls(home)

    @classmethod
    def discover(cls, start: str | os.PathLike[str] | None = None) -> "Workspace":
        """The nearest experiment directory at or above ``start``, else the home.

        Standing IN an experiment folder still wins, because that is a real
        thing people do and the nearest answer is the right one.  What changed
        is what happens when there is no such folder: the answer is one fixed
        place, not "wherever this was launched from".
        """

        current = Path(start or Path.cwd()).expanduser().resolve()
        for candidate in (current, *current.parents):
            if any((candidate / marker).exists() for marker in cls.MARKERS):
                return cls(candidate)
        return cls.default()


class ExperimentSession:
    """Devices, a signal plane, and the ability to take and keep a shot."""

    @classmethod
    def open(
        cls,
        workspace: Workspace | str | os.PathLike[str],
        *,
        apparatus: str | os.PathLike[str] | None = None,
        template: str | None = None,
        catalog: DeviceCatalogSnapshot | None = None,
    ) -> "ExperimentSession":
        """Start a session from a written-down apparatus, or from a template.

        This is where the composition root earns its name: it knows how to reach
        a pulse server, which zlc_atom deliberately does not, so it supplies the
        dialler while the configuration supplies the endpoint.
        """

        from zlc_atom.install import (
            discover_device_catalog,
            installation_config_from_template,
        )
        from zlc_atom.install.configuration import load_installation_config

        space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        snapshot = catalog if catalog is not None else discover_device_catalog()
        if template is not None:
            config = installation_config_from_template(snapshot, template)
        else:
            path = Path(apparatus) if apparatus is not None else space.apparatus
            if not path.is_file():
                raise FileNotFoundError(
                    f"no apparatus at {path}; pass template='virtual' to start from a template"
                )
            config = load_installation_config(path)

        return cls.from_config(space, config, catalog=snapshot)

    @classmethod
    def from_config(
        cls,
        workspace: Workspace | str | os.PathLike[str],
        config: object,
        *,
        catalog: DeviceCatalogSnapshot | None = None,
    ) -> "ExperimentSession":
        """Start from the exact DeviceManager draft accepted by its Init action."""

        from zlc_atom.install import discover_device_catalog
        from zlc_atom.install.configuration import InstallationConfig

        if not isinstance(config, InstallationConfig):
            raise TypeError("config must be InstallationConfig")
        space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        # Every experiment gets the packaged default pulses, not only the
        # checkout's home: calibration and the scan work out of the box in a
        # fresh folder, and the seeder never touches operator content.
        seed_packaged_pulses(space.pulses)
        snapshot = catalog if catalog is not None else discover_device_catalog()
        return cls._from_config(space, config, snapshot)

    @classmethod
    def _from_config(
        cls,
        space: Workspace,
        config: InstallationConfig,
        catalog: DeviceCatalogSnapshot,
    ) -> "ExperimentSession":
        from zlc_atom.install import create_installation
        from zlc_atom.install.configuration import InstallationConfig
        from zlc_runtime import SignalDataPlane

        if not isinstance(config, InstallationConfig):
            raise TypeError("config must be InstallationConfig")

        return cls(
            installation=create_installation(
                config.specs(),
                simulation=_resolved_simulation(space, config),
                catalog=catalog,
                connect_pulse=_connect_pulse,
            ),
            signal_plane=SignalDataPlane(),
            workspace=space,
            installation_config=config,
            device_catalog=catalog,
        )

    def __init__(
        self,
        *,
        installation: object,
        signal_plane: object,
        workspace: Workspace,
        installation_config: object | None = None,
        device_catalog: object | None = None,
    ) -> None:
        self.installation = installation
        self.signal_plane = signal_plane
        self.workspace = workspace.prepare()
        self.device_use = DeviceUseCoordinator()
        self._installation_config = installation_config
        self._device_catalog = device_catalog
        self._installation_revision = 0
        self._reconcile_lock = threading.RLock()
        self._recovery_installations: list[object] = []

    # ---------------------------------------------------------------- devices

    @property
    def camera(self) -> object:
        return self.installation.capability("camera.adapter")

    @property
    def sequencer(self) -> object:
        return self.installation.device("sequencer")

    @property
    def failures(self) -> Mapping[str, BaseException]:
        """Devices that did not open, by key.  An empty mapping means all did."""

        return self.installation.failures

    @property
    def installation_config(self) -> object | None:
        """The effective live declaration, including operational Close changes."""

        with self._reconcile_lock:
            return self._installation_config

    def _project_effective_installation_config(self) -> None:
        """Make the authored live baseline match the leaves still reachable."""

        from zlc_atom.install.configuration import InstallationConfig

        current = self._installation_config
        if not isinstance(current, InstallationConfig):
            return
        live = frozenset(self.installation.devices)
        self._installation_config = InstallationConfig(
            tuple(item for item in current.devices if item.instance_id in live),
            simulation=current.simulation,
        )
        self._installation_revision += 1

    def _close_recovery_installations(self) -> None:
        """Retry unusable successor leaves before another device transition."""

        for recovery in tuple(reversed(self._recovery_installations)):
            recovery.close()
            self._recovery_installations.remove(recovery)

    def plan_device_reconcile(
        self,
        config: object,
        *,
        close_keys: tuple[str, ...] | frozenset[str] = (),
    ) -> DeviceReconcilePlan:
        """Classify a live apparatus edit without touching a device.

        Identity is the authored ``instance_id``.  A role-only edit is metadata;
        a type/parameter edit replaces that leaf.  Dependants are rebuilt when
        the type they were constructed from changes.  A simulation-world edit
        rebuilds world-bound leaves and their dependants while independent
        physical leaves retain identity.
        """

        from zlc_atom.install import preflight_installation
        from zlc_atom.install.configuration import InstallationConfig

        if not isinstance(config, InstallationConfig):
            raise TypeError("config must be InstallationConfig")
        if not isinstance(close_keys, (tuple, frozenset)):
            raise TypeError("close_keys must be a tuple or frozenset")
        closed = frozenset(str(key) for key in close_keys)
        if any(not key for key in closed):
            raise ValueError("close_keys must contain non-empty device keys")
        with self._reconcile_lock:
            current = self._installation_config
            catalog = self._device_catalog
            source = self.installation
            session_revision = self._installation_revision
        if not isinstance(current, InstallationConfig) or catalog is None:
            raise RuntimeError(
                "this session was not opened from an installation configuration"
            )
        loaded_keys = frozenset(source.devices)
        unknown_close = closed - loaded_keys
        if unknown_close:
            raise KeyError(f"no loaded devices {sorted(unknown_close)}")
        target_blueprint = preflight_installation(
            config.specs(),
            **(
                {"world": source.world}
                if closed
                else {"simulation": _resolved_simulation(self.workspace, config)}
            ),
            catalog=catalog,
        )
        descriptors = {item.type_id: item for item in catalog.available}

        current_by_key = {item.instance_id: item for item in current.devices}
        wanted_by_key = {item.instance_id: item for item in config.devices}
        source_world = source.world
        target_world = target_blueprint.world
        if source_world is None or target_world is None:
            world_changed = source_world is not target_world
        else:
            before = getattr(source_world, "config", None)
            after = getattr(target_world, "config", None)
            world_changed = (
                source_world is not target_world
                and (before is None or after is None or before != after)
            )

        affected: set[str] = set(closed)
        affected_types: set[str] = set()
        all_keys = set(current_by_key) | set(wanted_by_key)
        if world_changed:
            # Only factories owned by the simulation world, then their normal
            # dependency closure below, must be rebuilt.  Independent physical
            # devices retain their exact object across a virtual-world edit.
            for key, item in {**current_by_key, **wanted_by_key}.items():
                descriptor = descriptors[item.type_id]
                if descriptor.world_config is not None:
                    affected.add(key)
                    affected_types.add(item.type_id)
        for key in all_keys:
            before = current_by_key.get(key)
            after = wanted_by_key.get(key)
            if before is None:
                continue
            if after is None or not _same_device_setup(before, after):
                affected.add(key)
                affected_types.add(before.type_id)
                if after is not None:
                    affected_types.add(after.type_id)
        for key in closed:
            affected_types.add(current_by_key[key].type_id)

        # Factories receive dependency leaves at construction time.  Reusing a
        # dependant after replacing its dependency would preserve a stale
        # object reference even when the logical keys look unchanged.
        affected.update(target_blueprint.dependent_keys(affected_types))

        # Close is an operational request, not an edit to apparatus.json.  Its
        # dependency closure is omitted from the live target; the untouched
        # draft remains available for an explicit Apply that opens it again.
        omitted = affected if closed else set()
        target = InstallationConfig(
            tuple(item for item in config.devices if item.instance_id not in omitted),
            simulation=config.simulation,
        )
        target_by_key = {item.instance_id: item for item in target.devices}
        retained = tuple(
            key
            for key in source.devices
            if key in target_by_key
            and key not in affected
            and key in current_by_key
            and _same_device_setup(current_by_key[key], target_by_key[key])
        )
        build = tuple(
            item.instance_id
            for item in target.devices
            if item.instance_id not in retained
        )
        affected_keys = tuple(
            dict.fromkeys(
                (*tuple(key for key in source.devices if key not in retained), *build)
            )
        )
        return DeviceReconcilePlan(
            source_installation=source,
            source_revision=int(source.revision),
            session_revision=session_revision,
            target_config=target,
            retained_keys=retained,
            affected_keys=affected_keys,
            build_keys=build,
            world_changed=world_changed,
        )

    def reconcile_devices(self, plan: DeviceReconcilePlan) -> None:
        """Apply one admitted plan while preserving every unchanged leaf."""

        from zlc_atom.install import (
            Installation,
            InstallationCompositionError,
            create_installation,
            preflight_installation,
        )
        from zlc_atom.install.configuration import InstallationConfig

        if not isinstance(plan, DeviceReconcilePlan):
            raise TypeError("plan must be DeviceReconcilePlan")
        target_config = plan.target_config
        if not isinstance(target_config, InstallationConfig):
            raise TypeError("plan target_config must be InstallationConfig")
        with self._reconcile_lock:
            if self.installation is not plan.source_installation:
                raise RuntimeError("live installation changed after reconcile planning")
            if self._installation_revision != plan.session_revision:
                raise RuntimeError("session installation revision changed")
            source = self.installation
            if int(source.revision) != plan.source_revision:
                raise RuntimeError("installation ownership revision changed")
            self._close_recovery_installations()

            # Metadata-only edits need no ownership transition at all.
            if not plan.affected_keys and not plan.build_keys:
                self._installation_config = target_config
                self._installation_revision += 1
                return

            holding = Installation(
                {},
                world=source.world,
                broker=source.broker,
            )
            if plan.retained_keys:
                source.transfer_leaves_to(
                    holding,
                    list(plan.retained_keys),
                    source_revision=plan.source_revision,
                    target_revision=holding.revision,
                )
            build_by_key = {
                item.instance_id: item
                for item in target_config.devices
                if item.instance_id in plan.build_keys
            }
            successor_blueprint = None
            if build_by_key or plan.world_changed:
                build_specs = tuple(
                    {
                        "key": item.instance_id,
                        "type_id": item.type_id,
                        "config": item.to_dict()["parameters"],
                    }
                    for item in target_config.devices
                    if item.instance_id in build_by_key
                )
                catalog = self._device_catalog
                assert catalog is not None
                preflight_kwargs = {
                    "catalog": catalog,
                    "borrowed_from": holding,
                    "borrowed_revision": holding.revision,
                }
                if plan.world_changed:
                    preflight_kwargs["simulation"] = _resolved_simulation(
                        self.workspace, target_config
                    )
                else:
                    preflight_kwargs["world"] = holding.world
                try:
                    successor_blueprint = preflight_installation(
                        build_specs,
                        **preflight_kwargs,
                    )
                except BaseException:
                    # No closer has run yet.  Restore the one ownership root so
                    # a failed internal successor preflight changes no live
                    # session reachability.
                    if plan.retained_keys:
                        holding.transfer_leaves_to(
                            source,
                            list(plan.retained_keys),
                            source_revision=holding.revision,
                            target_revision=source.revision,
                        )
                    raise
            # Make every retained leaf continuously reachable while affected
            # devices close and replacements start on the worker thread.
            self.installation = holding
            try:
                source.close()
            except BaseException as close_error:
                # A failed closer remains owned and bound by source.  Move it
                # back beside the retained leaves before reporting failure so
                # the live session never loses a device that is still open.
                remaining = tuple(source.devices)
                if remaining:
                    source.transfer_leaves_to(
                        holding,
                        list(remaining),
                        source_revision=source.revision,
                        target_revision=holding.revision,
                    )
                self._project_effective_installation_config()
                raise close_error

            if successor_blueprint is None:
                self._installation_config = target_config
                self._installation_revision += 1
                return
            try:
                successor = create_installation(
                    successor_blueprint,
                    broker=holding.broker,
                    connect_pulse=_connect_pulse,
                )
            except BaseException as create_error:
                if isinstance(create_error, InstallationCompositionError):
                    self._recovery_installations.append(create_error.recovery)
                self._project_effective_installation_config()
                raise
            try:
                retained_now = tuple(holding.devices)
                if retained_now:
                    holding.transfer_leaves_to(
                        successor,
                        list(retained_now),
                        source_revision=holding.revision,
                        target_revision=successor.revision,
                    )
            except BaseException as transfer_error:
                try:
                    successor.close()
                except BaseException as close_error:
                    self._recovery_installations.append(successor)
                    self._project_effective_installation_config()
                    raise BaseExceptionGroup(
                        "device reconcile transfer and cleanup failed",
                        [transfer_error, close_error],
                    ) from None
                self._project_effective_installation_config()
                raise
            self.installation = successor
            self._installation_config = target_config
            self._installation_revision += 1

    def acquire_device_command(
        self, owner: object, label: str, key: str, device: object,
    ):
        """Acquire this session's command claim without leaking claim types."""

        return self.device_use.acquire_command(
            owner, label, (DeviceClaim(key, key, device),)
        )

    # ------------------------------------------------------------------ pulse

    def load_pulse(self, name: str) -> Mapping[str, Any]:
        """Compile and apply the named workspace ``zlc.pulse.v1`` JSON pulse."""

        path = self.workspace.pulse(name)
        if not path.is_file():
            raise FileNotFoundError(f"no pulse named {name!r} in {self.workspace.pulses}")
        from zlc_pulse import compile_sequence, resolve_api_parameters

        state = read_pulse(path)
        # API parameters resolve at their AUTHORED values -- the same thing
        # On Pulse means in the editor.  Without this, any template declaring
        # API parameters (the scan and imaging templates both do) could not be
        # loaded by name at all.
        sequence = resolve_api_parameters(state.sequence)
        board = self.sequencer.describe()
        if sequence.target != board.target:
            raise ValueError(
                f"pulse target {sequence.target!r} does not match the installed "
                f"sequencer target {board.target!r}"
            )
        program = compile_sequence(sequence, board.geometry, board.clock_hz)
        lease = self._acquire_pulse_device()
        try:
            self.sequencer.load(program, source=sequence)
        finally:
            lease.release()
        self._pulse_sequence = sequence
        self._pulse_path = path
        self._pulse = {"name": sequence.name}
        return self._pulse

    @property
    def pulse_sequence(self) -> object | None:
        return getattr(self, "_pulse_sequence", None)

    @property
    def pulse_path(self) -> Path | None:
        return getattr(self, "_pulse_path", None)

    # ------------------------------------------------------------------- shot

    def _acquire_pulse_device(self):
        return self.acquire_device_command(
            object(), "ExperimentSession pulse", "sequencer", self.sequencer,
        )

    def fire(self, shots: int = 1, timeout: float = 5.0) -> None:
        """Fire one finite execution and wait for its terminal report."""

        if isinstance(shots, bool) or not isinstance(shots, Integral):
            raise TypeError("shots must be an integer")
        count = int(shots)
        lease = self._acquire_pulse_device()
        try:
            self.sequencer.fire(cycles=count)
            report = self.sequencer.wait_done(float(timeout) * count)
            if report is None:
                raise TimeoutError(
                    f"{count} pulse cycle(s) did not report done within "
                    f"{float(timeout) * count:g}s"
                )
            if report.fault:
                raise RuntimeError(f"pulse execution failed: {report.fault}")
        except BaseException as error:
            try:
                self.sequencer.safe()
            except BaseException as safe_error:
                raise BaseExceptionGroup(
                    "pulse drive failed and the sequencer did not go safe",
                    [error, safe_error],
                ) from None
            lease.release()
            raise
        else:
            lease.release()

    # ---------------------------------------------------------------- keeping

    def day_folder(self) -> Path:
        """Where today's saved work lands.

        Asked, not computed: zlc_durable owns the rule that a day is the
        organising key, and a second place deciding it is a second answer.
        """

        return day_folder(self.workspace.data, _date.today())

    # ----------------------------------------------------------------- closing

    def close(self) -> None:
        with self._reconcile_lock:
            self.device_use.assert_idle()
            failures: list[BaseException] = []
            try:
                self._close_recovery_installations()
            except BaseException as error:
                failures.append(error)
            if not self._recovery_installations:
                try:
                    self.installation.close()
                except BaseException as error:
                    failures.append(error)
            close = getattr(self.signal_plane, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("experiment session close failed", failures)

    def __enter__(self) -> "ExperimentSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
