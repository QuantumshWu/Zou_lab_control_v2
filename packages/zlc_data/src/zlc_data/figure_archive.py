"""One portable file holding datasets and everything needed to explain them.

The arrays go in as arrays.  Everything else -- what the apparatus was doing,
which pulse drove it, which node computed it, how the panel was configured --
goes in as ONE JSON string under ``info``.

That single choice is what makes the archive outlive us.  ``np.load`` refuses to
unpickle by default, and rightly: an archive whose metadata is pickled Python
objects can only be read by the code that wrote it, and stops opening the day a
class is renamed.  JSON reads with the standard library, forever, from any
language.  The rule this format holds itself to is that a reader needs numpy
and nothing else::

    with np.load(path) as archive:                 # allow_pickle stays False
        info = json.loads(str(archive["info"]))
        frames = archive["frames"]

The FORMAT lives here, with the datasets it carries, because everyone who
writes one writes the same one: a notebook saving a figure, Panel Edit saving
what is on screen, a task saving the frames it just took.  Where such a file
LANDS -- under which day, with which unique name -- is policy, and belongs to
whoever is doing the saving.  Each section of ``info`` is produced by the
package that owns that subject; this module only assembles them.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import json
from typing import Any

import numpy as np

from .io import snapshot_from_manifest, snapshot_manifest
from .value import OwnedSnapshot


__all__ = [
    "FIGURE_SCHEMA",
    "figure_bytes",
    "read_archive",
    "read_dataset",
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


def figure_bytes(
    name: str,
    *,
    arrays: Mapping[str, np.ndarray | OwnedSnapshot],
    sections: Mapping[str, Any],
) -> bytes:
    """Encode one portable figure without deciding where it lives.

    ``arrays`` are stored as arrays; ``sections`` become the JSON ``info``.  Each
    section is named for the subject it describes -- ``provenance``, ``pulse``,
    ``panel``, ``run_chain`` -- and is produced by whichever package owns that
    subject.

    An entry may be a bare array or an ``OwnedSnapshot``.  A snapshot also
    records its identity in the ``dataset`` section, which is what lets the
    figure be REOPENED rather than merely read: bare arrays keep their numbers
    but lose their axes, units and revision, and what comes back can no longer
    be plotted the way it was plotted.

    Every writer shares this exact encoding.  Keeping one encoder prevents the
    notebook, the GUI and the tasks from growing three formats that merely use
    the same suffix.
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

    buffer = BytesIO()
    np.savez_compressed(buffer, **{_INFO_KEY: np.asarray(info), **stored})
    return buffer.getvalue()


def read_archive(path: object) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one archive back: its info document and its arrays.

    Deliberately written the way an outside reader would write it, with
    ``allow_pickle`` left at its safe default -- if this function ever needs
    more than numpy and the standard library, the archive has stopped being
    readable by anyone but us.
    """

    with np.load(path) as archive:
        if _INFO_KEY not in archive.files:
            raise ValueError(f"{path} carries no {_INFO_KEY} document")
        info = json.loads(str(archive[_INFO_KEY]))
        arrays = {name: archive[name] for name in archive.files if name != _INFO_KEY}
    return info, arrays


def read_dataset(
    info: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray | OwnedSnapshot],
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
