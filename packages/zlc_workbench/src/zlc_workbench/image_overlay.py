"""Compose an Image plot transaction from one exact Occupancy publication.

``SiteMap`` stays atom-domain data and ``ImagePointOverlay`` stays plot-domain
data.  This module is the deliberately small Workbench seam that joins the
two for a panel.  It never infers a grid and it never asks a current Logic row
which run produced a retained publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from zlc_atom.nodes.calibration import TrapCalibration
from zlc_data import SITE
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointMarker, PointStatus

from .logic import stable_signal_key


__all__ = ["ImageOverlayResolver", "ResolvedImagePresentation"]


_MODES = frozenset({"off", "centers", "occupancy"})


@dataclass(frozen=True, slots=True)
class ResolvedImagePresentation:
    """One plot input plus the exact annotation decision used to build it."""

    frame: ImageFrame
    requested_mode: str
    resolved_mode: str
    calibration_path: str | None = None
    note: str | None = None


def _calibration_path(publication: object) -> str | None:
    record = getattr(publication, "run_record", {})
    parameters = record.get("parameters", {}) if hasattr(record, "get") else {}
    candidate = parameters.get("calibration_path") if hasattr(parameters, "get") else None
    if candidate is None:
        return None
    text = str(candidate).strip()
    return text or None


def _one_site_row(value: object, count: int, name: str) -> np.ndarray | None:
    """Return one exact per-site row, or ``None`` for a multi-frame value.

    A zlc-data Dataset stores its point-table domain at array axis 1.  Site is
    therefore not guessed from an ndarray shape or assumed to be the last
    dimension.  Occupancy status is meaningful for one displayed frame; a
    finite stack with several status rows deliberately falls back to centers.
    """

    schema = getattr(value, "schema", None)
    table = getattr(schema, "point_table", None)
    columns = () if table is None else tuple(getattr(table, "columns", ()))
    if (
        table is None
        or int(getattr(table, "row_count", -1)) != count
        or not any(getattr(column, "role", None) == SITE for column in columns)
    ):
        raise ValueError(f"{name} does not carry the calibration site axis")
    array = np.asarray(getattr(value, "values"))
    if array.ndim < 2 or array.shape[1] != count:
        raise ValueError(f"{name} has an incompatible point-table array shape")
    rows = np.moveaxis(array, 1, -1).reshape((-1, count))
    return rows[0] if rows.shape[0] == 1 else None


class ImageOverlayResolver:
    """Resolve SiteMap markers for exact publications from one signal plane.

    Calibration files have unique workspace names and are immutable run
    artifacts, so caching the last loaded path is sufficient.  No digest,
    fingerprint, or file-version protocol is introduced.
    """

    def __init__(self) -> None:
        self._loaded_path: str | None = None
        self._loaded_calibration: TrapCalibration | None = None

    def _load(self, path: str) -> TrapCalibration:
        resolved = str(Path(path).expanduser().resolve())
        if resolved != self._loaded_path:
            self._loaded_calibration = TrapCalibration.load(resolved)
            self._loaded_path = resolved
        assert self._loaded_calibration is not None
        return self._loaded_calibration

    @staticmethod
    def _empty(
        snapshot: object,
        revision: int,
        *,
        requested: str,
        note: str | None = None,
    ) -> ResolvedImagePresentation:
        return ResolvedImagePresentation(
            ImageFrame(snapshot, ImagePointOverlay.empty(revision)),
            requested,
            "off",
            note=note,
        )

    def resolve(
        self,
        value: object,
        publication: object,
        *,
        mode: str,
        overlay_revision: int,
    ) -> ResolvedImagePresentation:
        """Build the atomic image+markers input for one panel update."""

        requested = str(mode).strip().lower()
        if requested not in _MODES:
            raise ValueError("site overlay must be off, centers, or occupancy")
        snapshot = getattr(value, "snapshot", None)
        if snapshot is None:
            raise TypeError("image overlay requires a signal value snapshot")
        exact = getattr(publication, "value", None)
        if not callable(exact) or exact(getattr(value, "name", "")) is not value:
            raise ValueError("image overlay value must belong to its exact publication")
        if requested == "off":
            return self._empty(snapshot, overlay_revision, requested=requested)

        record = getattr(publication, "run_record", {})
        node_id = record.get("node") if hasattr(record, "get") else None
        if not isinstance(node_id, str) or not node_id.strip():
            node_id = None
        frame_signal = (
            None if node_id is None else stable_signal_key(node_id, "frame_judged")
        )
        occupied_signal = (
            None if node_id is None else stable_signal_key(node_id, "occupied")
        )
        valid_signal = None if node_id is None else stable_signal_key(node_id, "valid")
        if (
            frame_signal is None
            or occupied_signal is None
            or valid_signal is None
            or getattr(value, "name", None) != frame_signal
        ):
            return self._empty(
                snapshot,
                overlay_revision,
                requested=requested,
                note="site overlay requires an Occupancy frame_judged signal",
            )

        path = _calibration_path(publication)
        if path is None:
            raise ValueError("Occupancy publication has no calibration_path")
        calibration = self._load(path)
        site_map = calibration.site_map
        model = calibration.readout_model
        base_valid = (
            np.asarray(site_map.valid_sites, dtype=bool)
            & np.asarray(model.usable_sites, dtype=bool)
            & np.isfinite(model.thresholds)
        )
        statuses = [
            PointStatus.UNKNOWN if is_valid else PointStatus.INVALID
            for is_valid in base_valid
        ]
        resolved = "centers"
        note = None

        if requested == "occupancy":
            occupied_value = exact(occupied_signal)
            valid_value = exact(valid_signal)
            if occupied_value is None or valid_value is None:
                raise ValueError("Occupancy publication lacks occupied/valid siblings")
            occupied = _one_site_row(occupied_value, site_map.n_sites, "occupied")
            valid = _one_site_row(valid_value, site_map.n_sites, "valid")
            if occupied is not None and valid is not None:
                occupied = np.asarray(occupied, dtype=bool)
                valid = np.asarray(valid, dtype=bool) & base_valid
                statuses = [
                    PointStatus.INVALID
                    if not is_valid
                    else PointStatus.OCCUPIED
                    if is_occupied
                    else PointStatus.EMPTY
                    for is_occupied, is_valid in zip(occupied, valid, strict=True)
                ]
                resolved = "occupancy"
            else:
                note = (
                    "occupancy statuses require one displayed frame; "
                    "showing measured centers for this multi-frame dataset"
                )

        markers = (
            PointMarker(
                str(site_id),
                float(center[0]),
                float(center[1]),
                status,
                str(site_id),
            )
            for site_id, center, status in zip(
                site_map.site_ids,
                site_map.centers_xy,
                statuses,
                strict=True,
            )
        )
        overlay = ImagePointOverlay.from_markers(overlay_revision, markers)
        return ResolvedImagePresentation(
            ImageFrame(snapshot, overlay),
            requested,
            resolved,
            str(Path(path).expanduser().resolve()),
            note,
        )
