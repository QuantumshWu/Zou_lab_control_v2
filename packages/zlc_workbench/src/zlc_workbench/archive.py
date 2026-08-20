"""Write the formal figure archive to the path selected by Panel Save.

``zlc_data.figure_archive`` owns the format.  Workbench supplies the panel's
typed datasets and records, then commits those bytes atomically to the explicit
Save As path.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    read_archive,
    read_dataset,
    write_figure_archive,
)
from zlc_durable import atomic_write_file


__all__ = [
    "FIGURE_SCHEMA",
    "read_archive",
    "read_dataset",
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
    atomic_write_file(
        target,
        lambda stream: write_figure_archive(
            stream,
            target.stem if name is None else str(name),
            arrays=arrays,
            sections=sections,
        ),
    )
    return target
