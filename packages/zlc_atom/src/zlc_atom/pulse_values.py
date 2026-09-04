"""The value sets an experiment is running today: API values, and config.

A bias code or a MOT duration measured once is a fact about the apparatus,
not about whichever pulse happened to be open when it was measured.  Until
now the only thing that could say what an API slot holds was the pulse file
itself, so recalibrating the field meant opening every pulse and retyping
three numbers -- and a pulse missed that round quietly ran the old ones.

The current set lives in one file beside the pulses.  Every pulse loaded
from a workspace picks it up, and a pulse that declares none of its ids is
simply left alone: the set is meant to be carried across pulses, most of
which declare only some of it.

Config values are the same idea one layer down.  Where an API value answers
"what does this run want", a config value answers "what is this apparatus
calibrated at" -- a channel delay, a DAC bias.  Those belong to the BOARD, so
their current set is loaded onto the sequencer once and fills every pulse it
compiles; the pulse says only which of its fields are config parameters.

The grammar is ``zlc_pulse``'s; what is here is where the files live and how
they are found.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from zlc_durable import atomic_write_bytes, readable_json_bytes
from zlc_pulse import (
    config_values_from_tree,
    config_values_to_tree,
)


#: Beside ``pulses/`` in the workspace, because these values belong to the
#: apparatus the pulses run on rather than to any one of them.
CONFIG_VALUES_DIRECTORY = "config_values"
#: The set a session loads by itself when it opens a board.  Others in the
#: folder are saved sets an operator loads deliberately.
CURRENT_CONFIG_VALUES = "current.json"


def read_config_values(path: str | Path) -> tuple[str, str, dict[str, tuple[float, str]]]:
    """``(name, source, {parameter_id: (value, unit)})`` from one saved set."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".json":
        raise ValueError(f"config values must be JSON: {source}")
    return config_values_from_tree(json.loads(source.read_text(encoding="utf-8")))


def write_config_values(
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
        readable_json_bytes(config_values_to_tree(entries, name=name, source=source)),
    )


__all__ = [
    "CONFIG_VALUES_DIRECTORY",
    "CURRENT_CONFIG_VALUES",
    "read_config_values",
    "write_config_values",
]
