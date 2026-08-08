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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as _date
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from zlc_durable import day_folder

from .archive import write_figure


__all__ = ["ExperimentSession", "Workspace", "read_pulse"]


def _connect_pulse(host: str, port: int, **kwargs: Any) -> object:
    """Reach a pulse server.  Imported here, and nowhere in the domain package."""

    from zlc_pulse import connect

    return connect(host, port, **kwargs)


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
        (home / "pulses").mkdir(parents=True, exist_ok=True)
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


def read_pulse(path: str | os.PathLike[str]) -> tuple[Any, Mapping[str, Any]]:
    """Build one pulse file and return the program and what it says about itself.

    Shared by the session, which loads the program into a sequencer, and by the
    editor, which wants the sequence behind it.  One reader means a window can
    never show a pulse assembled differently from the one that will be fired.
    """

    source = Path(path)
    spec = importlib.util.spec_from_file_location(f"_zlc_pulse_{source.stem}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"{source} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build = getattr(module, "build", None)
    if not callable(build):
        raise AttributeError(f"{source} defines no build()")
    return build()


class ExperimentSession:
    """Devices, a signal plane, and the ability to take and keep a shot."""

    @classmethod
    def open(
        cls,
        workspace: Workspace | str | os.PathLike[str],
        *,
        apparatus: str | os.PathLike[str] | None = None,
        template: str | None = None,
    ) -> "ExperimentSession":
        """Start a session from a written-down apparatus, or from a template.

        This is where the composition root earns its name: it knows how to reach
        a pulse server, which zlc_atom deliberately does not, so it supplies the
        dialler while the configuration supplies the endpoint.
        """

        from zlc_atom.install.configuration import load_installation_config

        space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        if template is not None:
            specs: object = template
        else:
            path = Path(apparatus) if apparatus is not None else space.apparatus
            if not path.is_file():
                raise FileNotFoundError(
                    f"no apparatus at {path}; pass template='virtual' to start from a template"
                )
            specs = load_installation_config(path).specs()

        return cls._from_specs(space, specs)

    @classmethod
    def from_config(
        cls,
        workspace: Workspace | str | os.PathLike[str],
        config: object,
    ) -> "ExperimentSession":
        """Start from the exact DeviceManager draft accepted by its Init action."""

        from zlc_atom.install.configuration import InstallationConfig

        if not isinstance(config, InstallationConfig):
            raise TypeError("config must be InstallationConfig")
        space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        return cls._from_specs(space, config.specs())

    @classmethod
    def _from_specs(cls, space: Workspace, specs: object) -> "ExperimentSession":
        from zlc_atom.install import create_installation
        from zlc_runtime import SignalDataPlane

        return cls(
            installation=create_installation(specs, connect_pulse=_connect_pulse),
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
        """Compile the named workspace pulse and apply it to the sequencer.

        Returns the pulse's own description of itself -- how many camera windows
        it opens, what each frame means -- which the caller needs to collect it
        correctly and the archive needs to explain it later.
        """

        path = self.workspace.pulses / f"{name}.py"
        if not path.is_file():
            raise FileNotFoundError(f"no pulse named {name!r} in {self.workspace.pulses}")
        from zlc_atom.nodes import arm_sequencer

        program, metadata = read_pulse(path)

        # Through the one rule.  This window and the calibration task each had
        # their own copy of "tell the device which line triggers the camera,
        # then load", which is how the two came to differ on the failure path.
        arm_sequencer(self.sequencer, program, metadata)
        self._pulse_sequence = metadata.get("sequence")
        self._pulse_path = path
        self._pulse = {"name": name, **{k: v for k, v in metadata.items() if k != "sequence"}}
        return self._pulse

    @property
    def pulse_sequence(self) -> object | None:
        return getattr(self, "_pulse_sequence", None)

    @property
    def pulse_path(self) -> Path | None:
        return getattr(self, "_pulse_path", None)

    # ------------------------------------------------------------------- shot

    def fire(self, shots: int = 1, timeout: float = 5.0) -> None:
        """Fire the loaded pulse, waiting for each shot to finish.

        Waiting is not optional: the board detects commands on a rising edge and
        drops a second fire issued while the first is still running, so a bare
        loop takes exactly one shot on hardware while a virtual sequencer
        accepts them all.
        """

        for shot in range(int(shots)):
            self.sequencer.fire()
            report = self.sequencer.wait_done(timeout)
            # The report is the point of waiting.  Throwing it away made a shot
            # that reported an error, underran, or never finished at all look
            # exactly like a clean one -- and whatever the camera did or did not
            # collect was published as though the pulse had run.
            if report is None:
                raise TimeoutError(
                    f"shot {shot + 1} of {int(shots)} did not report done within "
                    f"{timeout:g}s"
                )
            if report.fault:
                raise RuntimeError(f"shot {shot + 1} of {int(shots)}: {report.fault}")

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
        arrays: Mapping[str, np.ndarray],
        nodes: Sequence[object] = (),
        panel: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save arrays together with everything needed to explain them.

        Each section comes from whoever owns that subject: the nodes describe the
        apparatus and what was asked of it, the pulse describes the stimulus.
        This assembles them; it does not author them.
        """

        provenance: dict[str, Any] = {}
        for node in nodes:
            recorder = getattr(node, "provenance", None)
            capture = getattr(recorder, "capture", None)
            if callable(capture):
                record = capture(node)
                provenance[str(record.get("node", "node"))] = record

        sections: dict[str, Any] = {}
        if provenance:
            sections["provenance"] = provenance
        pulse = getattr(self, "_pulse", None)
        if pulse is not None:
            sections["pulse"] = pulse
        if panel:
            sections["panel"] = dict(panel)

        return write_figure(self.workspace.data, name, arrays=arrays, sections=sections)

    # ----------------------------------------------------------------- closing

    def close(self) -> None:
        self.installation.close()
        close = getattr(self.signal_plane, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "ExperimentSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
