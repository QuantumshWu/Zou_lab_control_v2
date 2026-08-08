"""Single resolver for project-owned pulse definition modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def arm_sequencer(sequencer: object, program: object, metadata: Mapping[str, Any]) -> None:
    """Tell the device which line triggers the camera, then load the program.

    One rule.  A pulse declares its trigger channel by board signal name, and a
    device that was not told carries an invented default -- so renaming a
    channel in the pulse produces a wrong frame count rather than an error.
    Every caller that loads a pulse has to do this, and two of them each had
    their own copy of it.
    """

    channel = metadata.get("camera_trigger_channel")
    if channel is not None and hasattr(sequencer, "camera_trigger_channel"):
        sequencer.camera_trigger_channel = str(channel)
    sequencer.load(program)


@dataclass(frozen=True)
class ResolvedPulse:
    """A built pulse and the provenance needed by an orchestrating task."""

    name: str
    path: Path | None
    program: object
    metadata: Mapping[str, Any]


def _normalize_result(name: str, path: Path | None, value: object) -> ResolvedPulse:
    if isinstance(value, tuple):
        if len(value) != 2 or not isinstance(value[1], Mapping):
            raise TypeError("pulse build() tuple result must be (program, metadata)")
        program, metadata = value
    else:
        program, metadata = value, {}
    if program is None:
        raise ValueError("pulse build() returned no program")
    return ResolvedPulse(name, path, program, dict(metadata))


def _load_module(path: Path, name: str) -> ModuleType:
    module_name = f"_zlc_atom_pulse_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import specification for pulse {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_pulse(
    name: str,
    *,
    search_paths: Sequence[str | Path],
    override: object | None = None,
    build_parameters: Mapping[str, object] | None = None,
) -> ResolvedPulse:
    """Resolve one named ``<name>.py`` pulse definition without fallback.

    The paths are the directories to look in, which is what they are called.
    They used to be workspace ROOTS with ``pulses`` appended here, so every
    caller had to know a subdirectory name this signature never mentions --
    and the one caller that read the parameter as written passed the pulse
    directory and got ``pulses/pulses/<name>.py``.  Where an experiment keeps
    its pulses is the workspace's fact to state, once.
    """

    pulse_name = str(name).strip()
    if not pulse_name or Path(pulse_name).name != pulse_name or pulse_name.endswith(".py"):
        raise ValueError("pulse name must be a plain module name without path or .py")
    parameters = dict(build_parameters or {})
    if override is not None:
        value = override(**parameters) if callable(override) else override
        return _normalize_result(pulse_name, None, value)

    if isinstance(search_paths, (str, Path)):
        search_paths = (search_paths,)
    roots = tuple(Path(value).expanduser().resolve() for value in search_paths)
    candidates = tuple(root / f"{pulse_name}.py" for root in roots)
    for path in candidates:
        if not path.is_file():
            continue
        module = _load_module(path, pulse_name)
        build = getattr(module, "build", None)
        if not callable(build):
            raise TypeError(f"pulse module {path} must export callable build()")
        return _normalize_result(pulse_name, path, build(**parameters))
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"pulse {pulse_name!r} was not found; searched these paths:\n{attempted}"
    )


__all__ = [
    "arm_sequencer","ResolvedPulse", "resolve_pulse"]
