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
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from zlc_data import OwnedSnapshot
# The identity projection is not part of zlc_data's notebook-facing facade: it
# is how a package embeds datasets in a composite file, which is exactly what
# this is.
from zlc_data import snapshot_from_manifest, snapshot_manifest
from zlc_durable import atomic_write_bytes, day_folder, unique_path


__all__ = [
    "FIGURE_SCHEMA",
    "read_archive",
    "read_dataset",
    "write_figure",
    "write_figure_file",
]


#: Bumped when the layout of ``info`` changes in a way a reader must notice.
FIGURE_SCHEMA = "zlc.figure/v1"

_INFO_KEY = "info"


def _jsonable(value: Any) -> Any:
    """Plain data a JSON reader can take, with numpy scalars unwrapped."""

    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError(
            "arrays belong beside info as their own entries, not inside it; "
            "pass them in `arrays` so a reader can memory-map and inspect them"
        )
    return str(value)


def _figure_bytes(
    name: str,
    *,
    arrays: Mapping[str, np.ndarray],
    sections: Mapping[str, Any],
) -> bytes:
    """Encode one portable figure without deciding where it lives.

    ``arrays`` are stored as arrays; ``sections`` become the JSON ``info``.  Each
    section is named for the subject it describes -- ``provenance``, ``pulse``,
    ``panel``, ``dataset`` -- and is produced by whichever package owns that
    subject.

    An entry may be a bare array or an ``OwnedSnapshot``.  A snapshot also
    records its identity in the ``dataset`` section, which is what lets the
    figure be REOPENED rather than merely read: bare arrays keep their numbers
    but lose their axes, units and revision, and what comes back can no longer
    be plotted the way it was plotted.  zlc_data produces that record, because
    what a dataset's identity IS belongs to zlc_data.

    ``write_figure`` and Panel Edit's explicit Save As path share this exact
    encoding.  Keeping one encoder prevents the notebook and GUI archives from
    becoming two formats that merely use the same suffix.
    """

    if not arrays:
        raise ValueError("a figure archive must carry at least one array")
    if _INFO_KEY in arrays:
        raise ValueError(f"{_INFO_KEY!r} is reserved for the metadata document")

    stored: dict[str, np.ndarray] = {}
    datasets: dict[str, Any] = {}
    for key, value in arrays.items():
        if isinstance(value, OwnedSnapshot):
            manifest = snapshot_manifest(
                value,
                stored,
                values_key=str(key),
                validity_key=f"{key}.validity",
            )
            # The full schema is already present beside the revision ref, so
            # persisting its derived digest again adds no scientific data.
            ref = dict(manifest["ref"])
            ref.pop("schema_fingerprint", None)
            manifest["ref"] = ref
            datasets[str(key)] = manifest
        else:
            stored[str(key)] = np.asarray(value)
    sections = dict(sections)
    if datasets:
        if "dataset" in sections:
            raise ValueError("the dataset section is written from the snapshots themselves")
        sections["dataset"] = datasets

    info = json.dumps(
        {"schema": FIGURE_SCHEMA, "name": str(name), "sections": _jsonable(sections)},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )

    from io import BytesIO

    buffer = BytesIO()
    np.savez_compressed(buffer, **{_INFO_KEY: np.asarray(info), **stored})
    return buffer.getvalue()


def write_figure_file(
    path: str | os.PathLike[str],
    *,
    arrays: Mapping[str, np.ndarray],
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
        _figure_bytes(
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
    arrays: Mapping[str, np.ndarray],
    sections: Mapping[str, Any],
    when: _date | None = None,
) -> Path:
    """Write one uniquely named figure archive and return where it landed.

    The file is written atomically, so an interrupted save leaves the previous
    archive intact rather than a truncated one.
    """

    folder = day_folder(save_root, _date.today() if when is None else when)
    path = unique_path(folder, name, ".npz")
    atomic_write_bytes(
        path,
        _figure_bytes(name, arrays=arrays, sections=sections),
    )
    return path


def read_archive(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one archive back: its info document and its arrays.

    Deliberately written the way an outside reader would write it, with
    ``allow_pickle`` left at its safe default -- if this function ever needs
    more than numpy and the standard library, the archive has stopped being
    readable by anyone but us.
    """

    with np.load(Path(path)) as archive:
        if _INFO_KEY not in archive.files:
            raise ValueError(f"{path} carries no {_INFO_KEY} document")
        info = json.loads(str(archive[_INFO_KEY]))
        arrays = {name: archive[name] for name in archive.files if name != _INFO_KEY}
    return info, arrays


def read_dataset(
    info: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    name: str,
) -> OwnedSnapshot:
    """Rebuild one saved dataset, axes and all.

    The reason the identity was written: this returns what was saved, not a
    lookalike assembled from the numbers.  A figure that can only be re-read as
    an array cannot be re-plotted, re-fitted, or compared with a later run.
    """

    manifests = info.get("sections", {}).get("dataset", {})
    manifest = manifests.get(str(name))
    if manifest is None:
        raise KeyError(
            f"{name!r} was saved as a bare array, so its axes were not recorded"
        )
    return snapshot_from_manifest(manifest, arrays)
