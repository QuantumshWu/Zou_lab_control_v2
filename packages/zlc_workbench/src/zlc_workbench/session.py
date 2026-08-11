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
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from zlc_data import OwnedSnapshot
from zlc_durable import atomic_write_bytes, day_folder, readable_json_bytes

from .archive import write_figure
from .device_use import DeviceClaim, DeviceUseCoordinator
from .pulse_state import PulseEditorState, read_pulse

if TYPE_CHECKING:
    from zlc_atom.install import DeviceCatalogSnapshot
    from zlc_atom.install.configuration import InstallationConfig


__all__ = ["ExperimentSession", "PulseEditorState", "Workspace", "read_pulse"]


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
    )

    pulses.mkdir(parents=True, exist_ok=True)
    _seed_default_pulse(
        pulses / Workspace.IMAGING_TEMPLATE, calibration_pulse_template_bytes()
    )
    _seed_default_pulse(
        pulses / "mot_field_template.json", scan_pulse_template_bytes()
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
                catalog=catalog,
                connect_pulse=_connect_pulse,
            ),
            signal_plane=SignalDataPlane(),
            workspace=space,
        )

    def __init__(
        self,
        *,
        installation: object,
        signal_plane: object,
        workspace: Workspace,
    ) -> None:
        self.installation = installation
        self.signal_plane = signal_plane
        self.workspace = workspace.prepare()
        self.device_use = DeviceUseCoordinator()

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
        return self.device_use.acquire_command(
            object(),
            "ExperimentSession pulse",
            (
                DeviceClaim(
                    "sequencer",
                    "sequencer",
                    self.sequencer,
                    "exclusive",
                ),
            ),
        )

    def fire(self, shots: int = 1, timeout: float = 5.0) -> None:
        """Fire the loaded pulse, waiting for each shot to finish.

        Waiting is not optional: the board detects commands on a rising edge and
        drops a second fire issued while the first is still running, so a bare
        loop takes exactly one shot on hardware while a virtual sequencer
        accepts them all.
        """

        count = int(shots)
        lease = self._acquire_pulse_device()
        try:
            for shot in range(count):
                self.sequencer.fire()
                report = self.sequencer.wait_done(timeout)
                # The report is the point of waiting.  Throwing it away made a shot
                # that reported an error, underran, or never finished at all look
                # exactly like a clean one -- and whatever the camera did or did not
                # collect was published as though the pulse had run.
                if report is None:
                    raise TimeoutError(
                        f"shot {shot + 1} of {count} did not report done within "
                        f"{timeout:g}s"
                    )
                if report.fault:
                    raise RuntimeError(f"shot {shot + 1} of {count}: {report.fault}")
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

    def save_figure(
        self,
        name: str,
        *,
        arrays: Mapping[str, np.ndarray | OwnedSnapshot],
        panel: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save arrays together with everything needed to explain them.

        Typed snapshots carry their own dataset identity; the current pulse
        describes the stimulus.  Panel Edit's dedicated save path additionally
        records the selected publication's run chain.
        """

        sections: dict[str, Any] = {}
        pulse = getattr(self, "_pulse", None)
        if pulse is not None:
            sections["pulse"] = pulse
        if panel:
            sections["panel"] = dict(panel)

        return write_figure(self.workspace.data, name, arrays=arrays, sections=sections)

    # ----------------------------------------------------------------- closing

    def close(self) -> None:
        self.device_use.assert_idle()
        self.installation.close()
        close = getattr(self.signal_plane, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "ExperimentSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
