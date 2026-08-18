"""Saving a figure with everything needed to explain it.

One file per saved figure, filed under the day it was taken:

    <save_root>/2026_08_05/mot-loading.npz

The arrays go in as arrays.  Everything else -- what the apparatus was doing,
which pulse drove it, which node computed it, how the panel was configured --
goes in as ONE JSON string under ``info``.

That single choice is what makes the archive outlive us.  ``np.load`` refuses to
unpickle by default, and rightly: an archive whose metadata is pickled Python
objects can only be read by the code that wrote it, and stops opening the day a
class is renamed.  JSON reads with the standard library, forever, from any
language.  The rule this package holds itself to is that a reader needs numpy
and nothing else:

    with np.load(path) as archive:                 # allow_pickle stays False
        info = json.loads(str(archive["info"]))
        frames = archive["frames"]

The composition root writes archives because it is the only place that can see
an experiment whole.  It does NOT invent their contents: each section is
produced by the package that owns that subject, and this module only assembles
the sections and puts the file on disk.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date as _date
import os
from pathlib import Path
from typing import Any

import numpy as np
from zlc_data import OwnedSnapshot
# The FORMAT is zlc_data's -- it is a file of datasets, and everyone who writes
# one writes the same one.  What this module owns is WHERE such a file lands:
# under the day it was taken, with a name nothing else has.
from zlc_data.figure_archive import (
    FIGURE_SCHEMA,
    figure_bytes,
    read_archive,
    read_dataset,
)
from zlc_durable import atomic_write_bytes, day_folder, unique_path


__all__ = [
    "FIGURE_SCHEMA",
    "read_archive",
    "read_dataset",
    "write_figure",
    "write_figure_file",
]


def write_figure_file(
    path: str | os.PathLike[str],
    *,
    arrays: Mapping[str, np.ndarray | OwnedSnapshot],
    sections: Mapping[str, Any],
    name: str | None = None,
) -> Path:
    """Write one archive to an explicit Panel Edit Save As path."""

    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        target,
        figure_bytes(
            target.stem if name is None else str(name),
            arrays=arrays,
            sections=sections,
        ),
    )
    return target


def write_figure(
    save_root: str | os.PathLike[str],
    name: str,
    *,
    arrays: Mapping[str, np.ndarray | OwnedSnapshot],
    sections: Mapping[str, Any],
    when: _date | None = None,
) -> Path:
    """Write one uniquely named figure archive and return where it landed.

    The file is written atomically, so an interrupted save leaves the previous
    archive intact rather than a truncated one.
    """

    folder = day_folder(save_root, _date.today() if when is None else when)
    return unique_path(
        folder,
        name,
        ".npz",
        writer=lambda temporary: temporary.write_bytes(
            figure_bytes(name, arrays=arrays, sections=sections)
        ),
    )
