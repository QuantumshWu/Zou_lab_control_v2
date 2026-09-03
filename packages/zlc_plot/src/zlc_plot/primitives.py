"""Domain-neutral overlays and inputs for specialised plot kinds.

Application code projects point annotations and pulse programs into these
records.  The plotting package never imports those domain layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from zlc_data import (
    AxisId,
    AxisSpec,
    OwnedSnapshot,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_data.snapshot_projection import (
    selection_indices,
    value_selection,
)

from .data_contract import resolve_axis, snapshot_revision
from .kinds import AxisDomain, AxisRef

from ._validation import finite_real as _finite
from ._validation import integer


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _nonnegative_time(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _scan_number(
    value: int | None,
    name: str,
    *,
    optional: bool,
) -> int | None:
    return integer(value, name, minimum=1, optional=optional)


class PointStatus(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    INVALID = "invalid"


IMAGE_POINT_OVERLAY_CONTRACT = "zlc_plot.image-point-overlay-status"
IMAGE_POINT_OVERLAY_GEOMETRY_RECORD = "image_point_overlay_geometry"


def _image_axes(snapshot: OwnedSnapshot) -> tuple[object, object]:
    axes = tuple(snapshot.block.schema.cell_domain.axes)
    if len(axes) != 2 or tuple(axis.role for axis in axes) != (
        SPATIAL_Y,
        SPATIAL_X,
    ):
        raise ValueError("image overlay geometry requires spatial-y, spatial-x")
    y_axis, x_axis = axes
    if y_axis.coordinate_frame != x_axis.coordinate_frame:
        raise ValueError("image overlay axes must share one coordinate frame")
    return y_axis, x_axis


def _axis_coordinates(axis: object, positions: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(
        tuple(axis.coordinate_at(index) for index in range(axis.size)),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("image overlay axis coordinates must be finite")
    if coordinates.size == 1:
        if not np.all(np.asarray(positions) == 0):
            raise ValueError(
                "a single-pixel image axis cannot infer an affine step"
            )
        return np.full(np.asarray(positions).shape, coordinates[0])
    differences = np.diff(coordinates)
    if not np.allclose(differences, differences[0]):
        raise ValueError("image overlay axes must be affine")
    step = float(differences[0])
    return float(coordinates[0]) + step * positions


def image_point_overlay_geometry(
    image: OwnedSnapshot,
    coordinates_xy: object,
    point_ids: object,
    *,
    status_axis: AxisSpec,
    labels: object | None = None,
    coordinates_are_indices: bool = False,
) -> dict[str, object]:
    """Build the strict plain geometry document paired with an overlay signal."""

    if not isinstance(image, OwnedSnapshot):
        raise TypeError("overlay image must be an OwnedSnapshot")
    if not isinstance(status_axis, AxisSpec):
        raise TypeError("overlay status_axis must be an AxisSpec")
    if type(coordinates_are_indices) is not bool:
        raise TypeError("coordinates_are_indices must be bool")
    ids = tuple(str(value).strip() for value in tuple(point_ids))
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("overlay point ids must be unique non-empty text")
    centers = np.asarray(coordinates_xy, dtype=np.float64)
    if centers.shape != (len(ids), 2) or not np.all(np.isfinite(centers)):
        raise ValueError("overlay coordinates must have finite shape (points, 2)")
    resolved_labels = (
        tuple(str(index) for index in range(1, len(ids) + 1))
        if labels is None
        else tuple(str(value).strip() for value in tuple(labels))
    )
    if len(resolved_labels) != len(ids) or any(
        not value for value in resolved_labels
    ):
        raise ValueError("overlay labels must match point ids")
    if int(status_axis.size) != len(ids):
        raise ValueError("overlay status_axis must match point ids")
    y_axis, x_axis = _image_axes(image)
    if coordinates_are_indices:
        centers = np.column_stack(
            (
                _axis_coordinates(x_axis, centers[:, 0]),
                _axis_coordinates(y_axis, centers[:, 1]),
            )
        )
    return {
        "coordinate_frame": (
            None
            if x_axis.coordinate_frame is None
            else str(x_axis.coordinate_frame)
        ),
        "x_axis_id": str(x_axis.axis_id),
        "y_axis_id": str(y_axis.axis_id),
        "point_ids": list(ids),
        "coordinates_xy": centers.tolist(),
        "labels": list(resolved_labels),
        "status_axis_id": str(status_axis.axis_id),
        "status_coordinates": [
            status_axis.coordinate_at(index) for index in range(status_axis.size)
        ],
    }


def _validated_overlay_geometry(
    geometry: object,
    image: OwnedSnapshot,
    status: OwnedSnapshot,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...]]:
    if not isinstance(geometry, Mapping):
        raise TypeError("image overlay geometry must be a mapping")
    expected = {
        "coordinate_frame",
        "x_axis_id",
        "y_axis_id",
        "point_ids",
        "coordinates_xy",
        "labels",
        "status_axis_id",
        "status_coordinates",
    }
    if set(geometry) != expected:
        raise ValueError("image overlay geometry fields are not canonical")
    ids = tuple(str(value).strip() for value in tuple(geometry["point_ids"]))
    labels = tuple(str(value).strip() for value in tuple(geometry["labels"]))
    if (
        not ids
        or any(not value for value in ids)
        or len(set(ids)) != len(ids)
        or len(labels) != len(ids)
        or any(not value for value in labels)
    ):
        raise ValueError("image overlay geometry identities are invalid")
    centers = np.asarray(geometry["coordinates_xy"], dtype=np.float64)
    if centers.shape != (len(ids), 2) or not np.all(np.isfinite(centers)):
        raise ValueError("image overlay geometry coordinates are invalid")
    y_axis, x_axis = _image_axes(image)
    frame = (
        None if x_axis.coordinate_frame is None else str(x_axis.coordinate_frame)
    )
    if (
        geometry["coordinate_frame"] != frame
        or str(geometry["x_axis_id"]) != str(x_axis.axis_id)
        or str(geometry["y_axis_id"]) != str(y_axis.axis_id)
    ):
        raise ValueError("image overlay geometry differs from image axes")
    image_schema = image.block.schema
    status_schema = status.block.schema
    if (
        status_schema.repeat_domain != image_schema.repeat_domain
        or status_schema.point_domain != image_schema.point_domain
    ):
        raise ValueError("image overlay status leading geometry differs from image")
    status_axes = tuple(status_schema.cell_domain.axes)
    if len(status_axes) != 1:
        raise ValueError("image overlay status must declare one trailing axis")
    status_axis = status_axes[0]
    status_coordinates = tuple(
        status_axis.coordinate_at(index) for index in range(status_axis.size)
    )
    if (
        str(geometry["status_axis_id"]) != str(status_axis.axis_id)
        or tuple(geometry["status_coordinates"]) != status_coordinates
    ):
        raise ValueError("image overlay geometry differs from status axis")
    return ids, centers, labels


@dataclass(frozen=True, slots=True)
class ImagePointOverlay:
    """One immutable revision of canonical x/y annotations for an image.

    ``coordinates`` has shape ``(N, 2)`` and stores canonical x then y.  The
    optional metadata arrays, when present, must contain exactly ``N`` items.
    Ordering is stable within a revision; ``point_ids`` provide identity
    across revisions when an application needs it.  Monotonicity is enforced
    by the receiving plot session.

    A dynamic layer retains one canonical bool/numeric status Dataset.  Its
    values say EMPTY/OCCUPIED and its Dataset validity says whether each point
    status was measured.  The renderer applies the image PlotSpec's scope and
    facet to the shared leading axes and draws a judgement only when the
    surface names one exact ``(repeat, point)`` cell.  A pooled cell has no
    per-shot judgement and therefore stays UNKNOWN.

    Hand-authored/calibration markers have no run axes.  Their one immutable
    ``static_statuses`` vector is the other, mutually exclusive case.
    """

    revision: int
    coordinates: np.ndarray
    point_ids: tuple[str, ...] | None = None
    labels: tuple[str | None, ...] | None = None
    static_statuses: tuple[PointStatus, ...] | None = None
    status: OwnedSnapshot | None = None

    def __post_init__(self) -> None:
        revision = integer(self.revision, "revision", minimum=0)
        assert revision is not None
        coordinates = np.asarray(self.coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
            raise ValueError("coordinates must have shape (N, 2)")
        if coordinates.dtype.kind not in "biuf":
            raise TypeError("coordinates must be numeric")
        canonical = np.asarray(coordinates, dtype=np.float64)
        if not np.all(np.isfinite(canonical)):
            raise ValueError("coordinates must be finite")
        frozen = np.frombuffer(
            np.ascontiguousarray(canonical).tobytes(order="C"),
            dtype=np.float64,
        ).reshape(canonical.shape)
        count = int(frozen.shape[0])

        point_ids = self.point_ids
        if point_ids is not None:
            point_ids = tuple(
                _text(point_id, "point_id") for point_id in point_ids
            )
            if len(point_ids) != count:
                raise ValueError("point_ids must have one item per coordinate")
            if len(point_ids) != len(set(point_ids)):
                raise ValueError("point_ids must be unique")

        labels = self.labels
        if labels is not None:
            labels = tuple(
                None if label is None else _text(label, "point label")
                for label in labels
            )
            if len(labels) != count:
                raise ValueError("labels must have one item per coordinate")

        static = self.static_statuses
        if static is not None:
            static = tuple(static)
            if len(static) != count:
                raise ValueError(
                    "static_statuses must have one item per coordinate"
                )
            if any(not isinstance(status, PointStatus) for status in static):
                raise TypeError("static_statuses must contain PointStatus values")

        status = self.status
        if static is not None and status is not None:
            raise ValueError(
                "static statuses and a dynamic status snapshot are mutually exclusive"
            )
        if status is not None:
            self._validate_dynamic_status(status, count)

        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "coordinates", frozen)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "static_statuses", static)

    @staticmethod
    def _validate_dynamic_status(
        snapshot: object,
        count: int,
    ) -> None:
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("overlay status must be an OwnedSnapshot")
        axes = tuple(snapshot.block.schema.cell_domain.axes)
        if len(axes) != 1 or int(axes[0].size) != count:
            raise ValueError(
                f"overlay status must declare one complete point axis of {count} items"
            )
        values = np.asarray(snapshot.block.values)
        if values.dtype != np.dtype(np.bool_) and values.dtype.kind not in "iuf":
            raise TypeError("overlay status values must be bool or real numeric")
        valid_values = values[snapshot.expanded_validity()]
        if valid_values.size and not np.all(np.isin(valid_values, (0, 1))):
            raise ValueError("numeric overlay status values must be 0 or 1")

    @property
    def count(self) -> int:
        return int(self.coordinates.shape[0])

    def statuses_for(
        self,
        spec: object,
        facet_value: object | None,
    ) -> tuple[PointStatus, ...] | None:
        """Return one exact surface's site judgements, never a pooled guess."""

        if self.static_statuses is not None:
            return self.static_statuses
        if self.status is None:
            return None
        schema = self.status.block.schema
        terms: dict[AxisId, object] = {}
        for ref, coordinate in tuple(getattr(spec, "scope", ())):
            axis_id = self._leading_axis_id(schema, ref)
            if axis_id is None:
                return None
            terms[axis_id] = coordinate
        facet = getattr(spec, "facet", None)
        if facet is not None and facet_value is not None:
            axis_id = self._leading_axis_id(schema, facet)
            if axis_id is None:
                return None
            terms[axis_id] = facet_value

        if terms:
            selection = value_selection(schema, terms)
            repeats, points, _data = selection_indices(schema, selection)
        else:
            repeats = range(schema.repeat_domain.size)
            points = range(schema.point_domain.size)
        repeat_indices = tuple(repeats)
        point_indices = tuple(points)
        if len(repeat_indices) != 1 or len(point_indices) != 1:
            return None
        repeat = repeat_indices[0]
        point = point_indices[0]
        acquired = np.asarray(
            self.status.expanded_validity()[repeat, point],
            dtype=np.bool_,
        )
        if not bool(np.any(acquired)):
            return None
        values = np.asarray(
            self.status.block.values[repeat, point],
            dtype=np.bool_,
        )
        return tuple(
            (
                PointStatus.INVALID
                if not acquired[index]
                else (
                    PointStatus.OCCUPIED
                    if values[index]
                    else PointStatus.EMPTY
                )
            )
            for index in range(self.count)
        )

    @staticmethod
    def _leading_axis_id(
        schema: object,
        ref: object,
    ) -> AxisId | None:
        if not isinstance(ref, AxisRef):
            raise TypeError("overlay scope/facet axes must be AxisRef values")
        if ref.domain is AxisDomain.CELL_DATA:
            return None
        try:
            return resolve_axis(schema, ref).axis_id
        except KeyError:
            return None

    @classmethod
    def empty(cls, revision: int) -> "ImagePointOverlay":
        """Create a revisioned empty layer for an explicit clear operation."""

        return cls(revision=revision, coordinates=np.empty((0, 2), dtype=np.float64))

def image_point_overlay_from_signal(
    geometry: Mapping[str, object],
    status: OwnedSnapshot,
    image: OwnedSnapshot,
    *,
    revision: int,
) -> ImagePointOverlay:
    """Adapt one contract-matched numeric signal into an image overlay."""

    if not isinstance(status, OwnedSnapshot):
        raise TypeError("overlay status signal must be an OwnedSnapshot")
    ids, coordinates, labels = _validated_overlay_geometry(
        geometry,
        image,
        status,
    )
    return ImagePointOverlay(
        revision=revision,
        coordinates=coordinates,
        point_ids=ids,
        labels=labels,
        status=status,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ImageFrame:
    """Immutable image-data and point-overlay presentation transaction.

    ``snapshot`` owns frame ordering. ``overlay`` remains independently
    revisioned, so unchanged layers may be reused and clears use
    :meth:`ImagePointOverlay.empty` rather than an unversioned sentinel.
    """

    snapshot: OwnedSnapshot
    overlay: ImagePointOverlay

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
        if not isinstance(self.overlay, ImagePointOverlay):
            raise TypeError("overlay must be ImagePointOverlay")

    @property
    def revision(self) -> int:
        return snapshot_revision(self.snapshot)


@dataclass(frozen=True, slots=True)
class PulseChannel:
    channel_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _text(self.channel_id, "channel_id"))
        object.__setattr__(self, "label", _text(self.label, "channel label"))


@dataclass(frozen=True, slots=True)
class PulseBlock:
    channel_id: str
    start: float
    stop: float
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _text(self.channel_id, "channel_id"))
        start = _nonnegative_time(self.start, "pulse start")
        stop = _nonnegative_time(self.stop, "pulse stop")
        if stop <= start:
            raise ValueError("pulse stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "label", _text(self.label, "pulse label"))


SLOT_KINDS = ("scan", "api")


def _slot_kind(value: object) -> str:
    kind = str(value)
    if kind not in SLOT_KINDS:
        raise ValueError(f"slot kind must be one of {SLOT_KINDS}")
    return kind


@dataclass(frozen=True, slots=True)
class PulseScanRegion:
    """One highlighted scan interval with a positive integer badge.

    ``kind`` says who writes the slot -- a table that sweeps it, or a host
    that sets it a row at a time.  It is the same slot and the same badge; the
    drawing colours it so a glance answers which.
    """

    start: float
    stop: float
    number: int
    kind: str = "scan"

    def __post_init__(self) -> None:
        start = _nonnegative_time(self.start, "scan-region start")
        stop = _nonnegative_time(self.stop, "scan-region stop")
        if stop <= start:
            raise ValueError("scan-region stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(
            self,
            "number",
            _scan_number(self.number, "scan-region number", optional=False),
        )
        object.__setattr__(self, "kind", _slot_kind(self.kind))


@dataclass(frozen=True, slots=True)
class PulseAnalogTrace:
    name: str
    label: str
    minimum: float
    maximum: float
    starts: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "analog trace name"))
        object.__setattr__(self, "label", _text(self.label, "analog trace label"))
        minimum = _finite(self.minimum, "analog trace minimum")
        maximum = _finite(self.maximum, "analog trace maximum")
        if maximum <= minimum:
            raise ValueError("analog trace maximum must be greater than minimum")
        starts = tuple(
            _nonnegative_time(value, "analog trace start") for value in self.starts
        )
        values = tuple(_finite(value, "analog trace value") for value in self.values)
        if bool(starts) != bool(values):
            raise ValueError(
                "analog trace starts and values must either both be empty or both be present"
            )
        if values and len(starts) != len(values) + 1:
            raise ValueError("analog trace starts must contain one more item than values")
        if any(right <= left for left, right in zip(starts, starts[1:])):
            raise ValueError("analog trace starts must be strictly increasing")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PulseRepeatMarker:
    start: float
    stop: float
    label: str

    def __post_init__(self) -> None:
        start = _nonnegative_time(self.start, "repeat start")
        stop = _nonnegative_time(self.stop, "repeat stop")
        if stop <= start:
            raise ValueError("repeat stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "label", _text(self.label, "repeat label"))


@dataclass(frozen=True, slots=True)
class PulseDacScanSegment:
    trace_name: str
    start: float
    stop: float
    value: float
    number: int | None = None
    kind: str = "scan"

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_name", _text(self.trace_name, "DAC trace name"))
        start = _nonnegative_time(self.start, "DAC scan start")
        stop = _nonnegative_time(self.stop, "DAC scan stop")
        if stop <= start:
            raise ValueError("DAC scan stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "value", _finite(self.value, "DAC scan value"))
        object.__setattr__(
            self,
            "number",
            _scan_number(self.number, "DAC scan number", optional=True),
        )
        object.__setattr__(self, "kind", _slot_kind(self.kind))


@dataclass(frozen=True, slots=True)
class PulseTimelineData:
    channels: tuple[PulseChannel, ...]
    blocks: tuple[PulseBlock, ...]
    scan_regions: tuple[PulseScanRegion, ...] = ()
    time_unit: str = "s"
    total_duration: float | None = None
    analog_traces: tuple[PulseAnalogTrace, ...] = ()
    repeat_markers: tuple[PulseRepeatMarker, ...] = ()
    repeat_notation: str = ""
    scan_dac_segments: tuple[PulseDacScanSegment, ...] = ()

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("PulseTimeline requires at least one channel")
        if any(not isinstance(item, PulseChannel) for item in channels):
            raise TypeError("channels must contain PulseChannel values")
        ids = tuple(item.channel_id for item in channels)
        if len(ids) != len(set(ids)):
            raise ValueError("pulse channel ids must be unique")
        blocks = tuple(self.blocks)
        if any(not isinstance(item, PulseBlock) for item in blocks):
            raise TypeError("blocks must contain PulseBlock values")
        unknown = {item.channel_id for item in blocks} - set(ids)
        if unknown:
            raise ValueError(f"pulse blocks reference unknown channels: {sorted(unknown)}")
        regions = tuple(self.scan_regions)
        if any(not isinstance(item, PulseScanRegion) for item in regions):
            raise TypeError("scan_regions must contain PulseScanRegion values")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "scan_regions", regions)
        object.__setattr__(self, "time_unit", _text(self.time_unit, "time_unit"))
        analog_traces = tuple(self.analog_traces)
        if any(not isinstance(item, PulseAnalogTrace) for item in analog_traces):
            raise TypeError("analog_traces must contain PulseAnalogTrace values")
        names = tuple(item.name for item in analog_traces)
        if len(names) != len(set(names)):
            raise ValueError("analog trace names must be unique")
        repeat_markers = tuple(self.repeat_markers)
        if any(not isinstance(item, PulseRepeatMarker) for item in repeat_markers):
            raise TypeError("repeat_markers must contain PulseRepeatMarker values")
        scan_dac_segments = tuple(self.scan_dac_segments)
        if any(not isinstance(item, PulseDacScanSegment) for item in scan_dac_segments):
            raise TypeError("scan_dac_segments must contain PulseDacScanSegment values")
        unknown_traces = {item.trace_name for item in scan_dac_segments} - set(names)
        if unknown_traces:
            raise ValueError(f"DAC scan segments reference unknown traces: {sorted(unknown_traces)}")
        scan_numbers = [item.number for item in regions]
        scan_numbers.extend(
            item.number for item in scan_dac_segments if item.number is not None
        )
        if len(scan_numbers) != len(set(scan_numbers)):
            raise ValueError("scan numbers must be unique across the timeline")
        total_duration = self.total_duration
        if total_duration is not None:
            total_duration = _finite(total_duration, "total_duration")
            if total_duration <= 0.0:
                raise ValueError("total_duration must be positive")
            latest = max(
                [0.0]
                + [item.stop for item in blocks]
                + [item.stop for item in regions]
                + [item.stop for item in repeat_markers]
                + [item.stop for item in scan_dac_segments]
                + [item.starts[-1] for item in analog_traces if item.starts]
            )
            if total_duration < latest:
                raise ValueError("total_duration cannot end before pulse content")
        object.__setattr__(self, "total_duration", total_duration)
        object.__setattr__(self, "analog_traces", analog_traces)
        object.__setattr__(self, "repeat_markers", repeat_markers)
        object.__setattr__(
            self,
            "repeat_notation",
            _text(self.repeat_notation, "repeat_notation"),
        )
        object.__setattr__(self, "scan_dac_segments", scan_dac_segments)


PlotInput: TypeAlias = OwnedSnapshot | ImageFrame | PulseTimelineData


__all__ = [
    "IMAGE_POINT_OVERLAY_CONTRACT",
    "IMAGE_POINT_OVERLAY_GEOMETRY_RECORD",
    "ImageFrame",
    "ImagePointOverlay",
    "PlotInput",
    "PointStatus",
    "image_point_overlay_from_signal",
    "image_point_overlay_geometry",
    "PulseBlock",
    "PulseAnalogTrace",
    "PulseChannel",
    "PulseDacScanSegment",
    "PulseRepeatMarker",
    "PulseScanRegion",
    "PulseTimelineData",
]
