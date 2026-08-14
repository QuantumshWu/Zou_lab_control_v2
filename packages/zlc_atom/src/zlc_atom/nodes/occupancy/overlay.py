"""Turn one occupancy judgement into the site rings drawn over its image.

This is DOMAIN knowledge and it lives beside the facts it needs: the site
GEOMETRY is a calibration fact, and the per-site, per-frame STATUS is the
occupancy result.  A notebook calls this exactly as the task console does --
the console holds no physics, so there is nowhere else for it to live.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from zlc_data import OwnedSnapshot, SITE
from zlc_plot import ImagePointOverlay, PointStatus

from .processor import OccupancyProcessor


#: Index order of the four judgements, so a NumPy reduction can carry them.
_STATUS_BY_CODE = (
    PointStatus.UNKNOWN,
    PointStatus.EMPTY,
    PointStatus.OCCUPIED,
    PointStatus.INVALID,
)

#: The two published outputs one overlay reads.  Named here because this
#: module owns the occupancy vocabulary; its callers hand over whatever the
#: node published and never learn which names matter.
_OCCUPIED = "occupied"
_VALID = "valid"


def _site_axed(snapshot: object, name: str, sites: int) -> np.ndarray:
    """Read one (repeat, frame, site) judgement, refusing anything else."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError(f"occupancy {name} must be a zlc_data.OwnedSnapshot")
    schema = snapshot.block.schema
    axes = tuple(schema.cell_schema.data_axes)
    if len(axes) != 1 or axes[0].role != SITE or int(axes[0].size) != sites:
        raise ValueError(
            f"occupancy {name} must declare one SITE cell axis of {sites} sites"
        )
    return np.asarray(snapshot.block.values, dtype=bool)


def _repeats_of(codes: np.ndarray) -> tuple[PointStatus, ...]:
    """One frame's judgement, taken across the repeats of that same frame.

    Repeats ARE the same question asked again, so a site they disagree about
    has no single answer and stays UNKNOWN.  Frames are not: they are
    different points of the pulse, and a picture that pooled them has no
    per-frame judgement to show at all -- which is why there is no row for
    the pooled case rather than an invented one.
    """

    flat = codes.reshape((-1, codes.shape[-1]))
    first = flat[0]
    agreed = np.where(np.all(flat == first, axis=0), first, 0)
    return tuple(_STATUS_BY_CODE[int(code)] for code in agreed)


def site_overlay(
    node: OccupancyProcessor,
    outputs: Mapping[str, OwnedSnapshot],
    *,
    revision: int,
) -> ImagePointOverlay:
    """Assemble the ring layer for the frames one occupancy node judged.

    ``node`` is the occupancy processor that produced ``outputs`` -- the
    authority on WHICH sites these are, since it holds the calibration whose
    site map is the only place their positions exist.  ``outputs`` is that
    node's published outputs by name, exactly as ``OccupancyResult.artifacts``
    spells them, so a notebook passes its result straight through.

    The returned overlay carries one geometry and one status row per FRAME
    coordinate of the cycle, keyed by the camera's own point column, plus the
    whole-picture row a standalone image draws.
    """

    if not isinstance(node, OccupancyProcessor):
        raise TypeError("site_overlay requires the occupancy node that judged")
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must map an occupancy output name to a snapshot")
    missing = tuple(name for name in (_OCCUPIED, _VALID) if name not in outputs)
    if missing:
        raise KeyError(f"occupancy overlay needs outputs {missing!r}")

    site_map = node.calibration.site_map
    sites = site_map.n_sites
    occupied = _site_axed(outputs[_OCCUPIED], _OCCUPIED, sites)
    valid = _site_axed(outputs[_VALID], _VALID, sites)
    if occupied.shape != valid.shape:
        raise ValueError("occupancy occupied and valid must share their shape")

    # INVALID outranks the judgement: a site the calibration cannot read has
    # no occupancy to report, and a ring that said EMPTY there would be a
    # measurement claim nobody made.
    codes = np.where(~valid, 3, np.where(occupied, 2, 1))
    frames = tuple(outputs[_OCCUPIED].block.schema.point_table.columns[0].values)
    statuses: dict[float | None, tuple[PointStatus, ...]] = {
        float(value): _repeats_of(codes[:, index])
        for index, value in enumerate(frames)
    }
    return ImagePointOverlay(
        revision=revision,
        coordinates=np.asarray(site_map.centers_xy, dtype=float),
        point_ids=tuple(site_map.site_ids),
        labels=tuple(str(index) for index in range(1, sites + 1)),
        statuses=statuses,
    )


__all__ = ["site_overlay"]
