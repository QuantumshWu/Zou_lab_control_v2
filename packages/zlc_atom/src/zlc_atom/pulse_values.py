"""The set of API parameter values an experiment is running today.

A bias code or a MOT duration measured once is a fact about the apparatus,
not about whichever pulse happened to be open when it was measured.  Until
now the only thing that could say what an API slot holds was the pulse file
itself, so recalibrating the field meant opening every pulse and retyping
three numbers -- and a pulse missed that round quietly ran the old ones.

The current set lives in one file beside the pulses.  Every pulse loaded
from a workspace picks it up, and a pulse that declares none of its ids is
simply left alone: the set is meant to be carried across pulses, most of
which declare only some of it.

The grammar is ``zlc_pulse``'s; what is here is where the file lives and how
a pulse finds it.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from zlc_durable import atomic_write_bytes, readable_json_bytes
from zlc_pulse import api_values_from_tree, api_values_to_tree


#: Beside ``pulses/`` in the workspace, because these values belong to the
#: apparatus the pulses run on rather than to any one of them.
API_VALUES_DIRECTORY = "api_values"
#: The set every pulse picks up by itself.  Others in the folder are saved
#: sets an operator loads deliberately.
CURRENT_API_VALUES = "current.json"


def read_api_values(path: str | Path) -> tuple[str, str, dict[str, tuple[float, str]]]:
    """``(name, source, {parameter_id: (value, unit)})`` from one saved set."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".json":
        raise ValueError(f"API values must be JSON: {source}")
    return api_values_from_tree(json.loads(source.read_text(encoding="utf-8")))


def write_api_values(
    path: str | Path,
    entries: Mapping[str, tuple[float, str]],
    *,
    name: str = "",
    source: str = "hand",
) -> None:
    """Write one saved set through the sole grammar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        target,
        readable_json_bytes(api_values_to_tree(entries, name=name, source=source)),
    )


def current_api_values_path(pulse_path: str | Path) -> Path:
    """Where the current set sits, for a pulse loaded from a workspace.

    A pulse resource is refused unless it sits directly in the workspace's
    ``pulses/`` folder, so the workspace is the folder holding that one -- the
    caller does not have to be handed a root it never asked for.
    """

    return (
        Path(pulse_path).expanduser().resolve().parent.parent
        / API_VALUES_DIRECTORY
        / CURRENT_API_VALUES
    )


__all__ = [
    "API_VALUES_DIRECTORY",
    "CURRENT_API_VALUES",
    "current_api_values_path",
    "read_api_values",
    "write_api_values",
]
